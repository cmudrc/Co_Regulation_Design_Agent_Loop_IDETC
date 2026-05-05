import os
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # design problem root (for design_problem_infra)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))  # project root (for system_infra)

from system_infra.auxiliary import generate_timestamped_basename, truncate_messages
from system_infra.model_adapters import LocalLlamaAdapter, LocalQwenAdapter, LMStudioAdapter, OpenAIAdapter, GoogleGeminiAdapter

from design_problem_infra.evaluation import detect_termination
from design_problem_infra.extra import generate_random_prompt, render_battery_pack
from design_problem_infra.tracker import step_conversion, DesignStep, DesignTrace
from design_problem_infra.validation import validate_design, print_validation_summary

from message.message_RWL import (
    design_agent_initial_message, 
    design_agent_follow_up_message
    )

# ==================== USER CONFIGURATION ====================
# Model configuration
MODEL_KEY_DESIGN_AGENT = "gemini-3.1-pro-preview"
# ============================================================

def main(problem_query=None, 
         required_specs=None, 
         verbose=True, 
         render=False, 
         output_dir="output", 
         experiment_label=None, 
         problem_seed=None, 
         model_seed=None):
    """
    Generate a single battery pack design for the given query.
    """

    max_steps = 30
    min_steps = 2
    step_count = 0

    design_agent_temperature = 1.0
    design_agent_max_tokens = 65536

    DesignAgent = GoogleGeminiAdapter(model_name=MODEL_KEY_DESIGN_AGENT, thinking_level="high", seed=model_seed)

    design_agent_messages = []
    is_valid = False
    is_done = False

    # initilize DesignTrace and Step
    step=DesignStep(step_number=step_count, 
                    text_response="",
                    is_valid=False)
    
    design_tree=DesignTrace(problem_query=problem_query,
                            steps=[],
                            required_specs=required_specs)
    design_tree.add_step(step)

    # populate experiment metadata
    design_tree.metadata["system_version"] = "RWL"
    design_tree.metadata["design_agent"] = {
        "model": DesignAgent.model_name,
        "thinking_level": DesignAgent.thinking_level,
        "seed": DesignAgent.seed,
        "temperature": design_agent_temperature,
    }
    if problem_seed is not None:
        design_tree.metadata["problem_seed"] = problem_seed

    # initialize design loop
    design_system_prompt = f"""
    You are an expert battery pack designer optimizing battery pack designs.
    """
    problem_prompt = design_agent_initial_message(problem_query, step)
    design_initial_msg = [
        {"role": "system", "content": design_system_prompt},
        {"role": "user", "content": problem_prompt}
        ]
    design_agent_messages.extend(design_initial_msg)
    
    while (is_valid == False or is_done == False or step_count < min_steps) and step_count < max_steps:
        step_count += 1

        # design agent generate "assistant" response
        design_response = DesignAgent.generate(design_agent_messages, temperature=design_agent_temperature, max_tokens=design_agent_max_tokens)
        design_response_text = design_response.text
        design_agent_messages.append({"role": "assistant", "content": design_response_text})

        # validate design
        design_dict, validation_results = validate_design(text=design_response_text, required_specs=required_specs)

        if verbose: 
            # print out design_dict and validation results
            print(f"\n[STEP {step_count}] Design Dictionary:")
            if design_dict:
                for key, value in design_dict.items():
                    if key != "cell_locations":  # Skip printing full cell_locations for readability
                        print(f"  {key}: {value}")
            print_validation_summary(validation_results)

        validatable = [
            result for key, result in validation_results.items()
            if hasattr(result, 'is_valid') and key != "extraction_error"
        ]
        is_valid = len(validatable) > 0 and all(r.is_valid for r in validatable)
        is_done = detect_termination(design_response_text)

        if step_count >= min_steps and is_valid and is_done: # if success, no feedback
            # update step
            step = step_conversion(step_number=step_count, 
                                   model_response=design_response, 
                                   design_dict=design_dict,
                                   validation_results=validation_results)
            design_tree.add_step(step)

        else:
            # update step
            step = step_conversion(step_number=step_count, 
                                   model_response=design_response, 
                                   design_dict=design_dict,
                                   validation_results=validation_results)
            design_tree.add_step(step)

            # update design agent message
            # generate "user" message to design agent
            design_msg_content = design_agent_follow_up_message(step)
            design_agent_messages.append({"role": "user", "content": design_msg_content})
 
    # Save message history regardless of outcome
    design_tree.add_message_history(design_agent_messages)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    if experiment_label:
        basename = experiment_label
    else:
        basename = generate_timestamped_basename()
    html_path = os.path.join(output_dir, f"design_tree_{basename}.html")
    json_path = os.path.join(output_dir, f"design_tree_{basename}.json")

    # Save the design trace as JSON
    design_tree.save_to_json(json_path)
    print(f"[INFO] Saved design trace to: {json_path}")

    if render:
        print(f"[INFO] Rendering battery pack to: {html_path}")
        if design_tree.final_design and design_tree.final_design.design_dict:
            cell_locations = design_tree.final_design.design_dict["cell_locations"]
            render_battery_pack(cell_locations, output_html=html_path, open_browser=False)
        else:
            print(f"[WARNING] No final design available for rendering")

    # If max retries reached, return None
    if step_count >= max_steps:
        print(f"[WARNING] No final design given by the agent within {max_steps} steps.")
        return None

    return design_tree


def batch_experiments(num_repeats=30, 
                      problem_seed=None, 
                      model_base_seed=15213, 
                      problem_query=None, 
                      required_specs=None, 
                      start_from=1, 
                      resume_dir=None):
    """Run batch experiments repeating the same prompt N times. Each experiment saves a JSON design trace.

    Args:
        num_repeats: Number of times to repeat the same prompt
        problem_seed: Seed used to generate the prompt (if problem_query is None)
        model_base_seed: Base seed from which per-repeat model seeds are derived
        problem_query: If provided, use this query directly instead of generating one
        required_specs: If provided alongside problem_query, use these specs directly
        start_from: Experiment number to start from (1-indexed). Use this to resume a crashed batch.
        resume_dir: If resuming, path to existing batch directory. If None, creates a new one.
    """

    # Resolve the single prompt to repeat
    if problem_query is not None:
        query = problem_query
    else:
        query, required_specs = generate_random_prompt(problem_seed)

    # Pre-generate all per-repeat model seeds from the base seed
    rng = random.Random(model_base_seed)
    model_seeds = [rng.randint(0, 2**31 - 1) for _ in range(num_repeats)]

    if resume_dir:
        batch_dir = resume_dir
    else:
        basename = generate_timestamped_basename()
        batch_dir = os.path.join(Path(__file__).parent.parent.parent, "output", f"RWL_RE_G31P_750_{basename}")
    os.makedirs(batch_dir, exist_ok=True)

    print(f"[INFO] Repeating prompt {num_repeats} times (seed={problem_seed})")
    print(f"[INFO] Model base seed: {model_base_seed} -> per-repeat seeds: {model_seeds[:5]}{'...' if num_repeats > 5 else ''}")
    print(f"[INFO] Prompt: {query}")
    print(f"[INFO] Results will be saved to: {batch_dir}")

    for i in range(start_from - 1, num_repeats):
        print(f"\n{'='*80}")
        print(f"Repeat {i+1}/{num_repeats} (model seed: {model_seeds[i]})")
        print(f"{'='*80}")

        # main loop (saves JSON per experiment)
        label = f"{problem_seed}_repeat_{i+1}"
        main(problem_query=query, 
             required_specs=required_specs, 
             verbose=False, 
             render=False, 
             output_dir=batch_dir, 
             experiment_label=label, 
             problem_seed=problem_seed, 
             model_seed=model_seeds[i])

    print(f"\n{'='*50}")
    print(f"[INFO] Batch experiments complete!")
    print(f"{'='*50}")


if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent.parent

    required_specs = {
            "voltage": 400,
            "capacity": 50,
            "current": 48,
            "temperature_C": 60,
            "width_mm": 1000,
            "depth_mm": 1000,
            "height_mm": 250,
            }
    
    problem_query = (
        f"Design a {required_specs['voltage']}V battery pack "
        f"with a minimum capacity of {required_specs['capacity']} Ah, "
        f"capable of continuously supplying at least {required_specs['current']}A of current draw "
        f"while staying at or below {required_specs['temperature_C']}C during operation, "
        f"within a {required_specs['width_mm']}mm × {required_specs['depth_mm']}mm × {required_specs['height_mm']}mm envelope."
    )

    batch_experiments(
        num_repeats=30,
        model_base_seed=15213, 
        problem_query=problem_query,
        required_specs=required_specs,
        start_from=1
    )

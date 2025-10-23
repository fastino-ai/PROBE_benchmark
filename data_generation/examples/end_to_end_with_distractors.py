import os
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from data_generation.data.linkedin_profile import LinkedInPersona
from data_generation.generators.world_model import (
    WorldModelGenerator,
    ContextDifficulty,
    DistractorDifficulty,
)
from data_generation.generators.bottleneck_injector import (
    BottleneckInjector,
)
from data_generation.generators.distractor_generator import DistractorGenerator
from data_generation.generators.multi_llm_client import create_default_multi_llm_client
from configs.unified_contracts import TruePositive


def main():
    # Configure logging for a clear step-by-step trace
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger = logging.getLogger("personasim.end_to_end")

    # Prepare run directory under generated_data
    base_dir = Path(__file__).parent / "generated_data" / "proactive_examples"
    run_dir = base_dir / datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    # File logger
    fh = logging.FileHandler(run_dir / "run.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logging.getLogger().addHandler(fh)

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found; set it before running.")
        return

    logger.info("[1/6] Building persona")
    persona = LinkedInPersona(
        name="Alex Johnson",
        occupation="Engineering Manager",
        location="Seattle, WA",
        about="Leads a team working on infra and platform reliability.",
    )
    (run_dir / "persona.json").write_text(
        json.dumps(persona.model_dump(), indent=2), encoding="utf-8"
    )

    logger.info("[2/6] Generating world model (this may take a bit)")
    world_model_generator = WorldModelGenerator()
    world_model = world_model_generator.generate(
        persona=persona, difficulty=ContextDifficulty.MEDIUM
    )
    logger.info(
        "World model ready: relationships=%d, actions=%d",
        len(world_model.relationships),
        len(world_model.available_actions),
    )
    (run_dir / "world_model.json").write_text(
        world_model.model_dump_json(indent=2), encoding="utf-8"
    )

    logger.info("[3/6] Injecting bottlenecks")
    injector = BottleneckInjector()
    bottlenecks = injector.inject_bottlenecks(
        world_model=world_model, difficulty=ContextDifficulty.MEDIUM
    )
    logger.info("Bottlenecks injected: %d", len(bottlenecks))
    try:
        b_list = [
            b.model_dump() if hasattr(b, "model_dump") else b for b in bottlenecks
        ]
    except Exception:
        b_list = [getattr(b, "__dict__", str(b)) for b in bottlenecks]
    (run_dir / "bottlenecks.json").write_text(
        json.dumps(b_list, indent=2), encoding="utf-8"
    )

    # Build a TP sketch for the first bottleneck (simple demo)
    logger.info("[4/6] Creating TruePositive sketch from first bottleneck (if any)")
    tp = TruePositive(
        target_problem=bottlenecks[0].description if bottlenecks else "",
        required_items=[],
        task_execution={
            "action_type": "send_followup",
            "target_artifacts": [],
            "execution_parameters": {},
            "expected_outcomes": [],
        },  # type: ignore
    )
    (run_dir / "tp.json").write_text(tp.model_dump_json(indent=2), encoding="utf-8")

    # LLM client wrapper
    logger.info("[5/6] Initializing LLM client and distractor generator")

    # Create MultiLLMClient and DistractorGenerator
    multi_llm_client = create_default_multi_llm_client()
    distractor_generator = DistractorGenerator(
        llm_generate_func=multi_llm_client.generate,
        debug_dir=run_dir / "distractors_debug",
    )

    logger.info("[6/6] Generating distractors using new interface")
    # Generate distractors using new interface - just pass the bottlenecks and let it handle the rest
    distractor_corpus_items = distractor_generator.generate(
        world_model=world_model,  # Use the WorldModel from generators, not the interface one
        bottlenecks=bottlenecks,  # Pass all bottlenecks
        k=6,  # Total distractors to generate
        difficulty=DistractorDifficulty.MEDIUM,
        actions=None,
    )

    # Convert to JSON format for saving
    distractors_json = [item.model_dump() for item in distractor_corpus_items]
    (run_dir / "distractors.json").write_text(
        json.dumps(distractors_json, indent=2), encoding="utf-8"
    )

    logger.info(
        "Distractors generated: total=%d",
        len(distractor_corpus_items),
    )
    print(f"Generated {len(distractor_corpus_items)} distractors")
    for i, d in enumerate(distractor_corpus_items, 1):
        why_not = d.metadata.get("why_not_satisfy", "Unrelated to bottleneck")
        print(f"{i}. type={d.type.value}, why_not='{why_not[:80]}'")

    print(f"Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()

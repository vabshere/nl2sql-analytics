# SQL Agent package

from dotenv import load_dotenv

load_dotenv()

from src.logging_config import configure_logging
from src.tracing import configure_tracing

configure_logging()
# WHY: configure_tracing reads OTLP_ENABLED from env (default false) so no
# API key or PipelineConfig is needed here. Any import of src — library mode,
# benchmark, scripts — gets the correct OTel provider automatically.
configure_tracing()

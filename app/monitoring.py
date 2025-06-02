import os
from typing import Callable

import numpy as np
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info

NAMESPACE = os.environ.get("METRICS_NAMESPACE", "fastapi")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "model")

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)


# ----- custom metrics -----
# def churn_confidence(
#     metric_name: str = "churn_confidence",
#     metric_doc: str = "Output churn confidence from classification model",
#     metric_namespace: str = "",
#     metric_subsystem: str = "",
# ) -> Callable[[Info], None]:
#     METRIC = Gauge(
#         metric_name,
#         metric_doc,
#         namespace=metric_namespace,
#         subsystem=metric_subsystem,
#     )

#     def instrumentation(info: Info) -> None:
#         if info.modified_handler == "/predict":
#             churn_confidence_val = info.response.headers.get("churn_probability")
#             if churn_confidence_val:
#                 METRIC.set(churn_confidence_val)

#     return instrumentation


# ----- add metrics -----
instrumentator.add(
    metrics.request_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.response_size(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.latency(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
instrumentator.add(
    metrics.requests(
        should_include_handler=True,
        should_include_method=True,
        should_include_status=True,
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

# buckets = (*np.arange(0, 10.5, 0.5).tolist(), float("inf"))
# instrumentator.add(
#     churn_confidence(metric_namespace=NAMESPACE, metric_subsystem=SUBSYSTEM)
# )
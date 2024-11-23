from typing import Any, Dict, Optional

from mmengine import Registry

IMPUTATIONS = Registry('imputations')


def build_imputation(
        cfg: Dict[str, Any], default_args: Optional[Dict[str, Any]] = None) -> Any:
    return IMPUTATIONS.build(cfg, default_args=default_args)

import yaml


class AttrDict(dict):
    """Dictionary with attribute-style access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def from_mapping(cls, value):
        if isinstance(value, dict):
            node = cls()
            for key, item in value.items():
                node[key] = cls.from_mapping(item)
            return node
        if isinstance(value, list):
            return [cls.from_mapping(item) for item in value]
        return value


def load_config(path="config.yml"):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader) or {}

    config.setdefault("out_dir", "./output/default_run")
    config.setdefault("cuda", True)
    config.setdefault("gpu_ids", [0])
    config.setdefault("manualSeed", 0)
    config.setdefault("threads", 0)

    return AttrDict.from_mapping(config)

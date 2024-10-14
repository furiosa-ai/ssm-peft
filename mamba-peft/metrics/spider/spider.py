
from metrics.spider.evaluation import build_foreign_key_map_from_json, evaluate
from utils.utils import flatten_dict



class SpiderMetric():
    def __init__(self) -> None:
        self.db_dir = "data/xlangai_spider/spider/database"
        self.etype = "all"
        self.table = "data/xlangai_spider/spider/tables.json"
        self.kmaps = build_foreign_key_map_from_json(self.table)

    def compute(self, predictions, references):
        out = evaluate(references, predictions, self.db_dir, self.etype, self.kmaps)
        out = flatten_dict(out, sep="/")
        out = {k: v for k, v in out.items() if not any(excl in k for excl in ["/partial", "/count"])}
        return out

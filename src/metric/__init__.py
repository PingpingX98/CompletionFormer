"""
    CompletionFormer
    ======================================================================

    BaseMetric implementation

    If you want to implement a new metric interface,
    it should inherit from the BaseMetric class.
"""


from importlib import import_module


def get(args):
    metric_name = args.model_name + 'Metric'
    module_name = 'metric.' + metric_name.lower()
    module = import_module(module_name)

    return getattr(module, metric_name)

# 对应新的图片保存格式
def getNew(args):
    metric_name = args.model_name + 'Metricnew'
    module_name = 'metric.' + metric_name.lower()
    module = import_module(module_name)

    return getattr(module, metric_name)

class BaseMetric:
    def __init__(self, args):
        self.args = args

    def evaluate(self, sample, output, mode):
        pass

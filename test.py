from resnet import run_model_cls
from huatu import huatu_cls


GPR_Frequency='300'
GPR_Csv="23057.csv"
res = run_model_cls(GPR_Frequency,GPR_Csv, "1.json")
#huatu_cls(300, 50, 0.06, 200, 180,100, GPR_Csv, res)

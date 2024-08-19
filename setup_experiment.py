 
import mlflow
import sys
print(sys.version)



print("MLflow version:", mlflow.__version__)

mlflow.set_experiment("my_ml_experiment")

#mlflow ui in new terminal R
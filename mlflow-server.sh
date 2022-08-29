#!/bin/bash

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns/

#!/bin/bash

pip install -r requirements.txt || echo "Failed to install requirements"

experiments/counterfactual.py

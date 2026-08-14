---
title: Predictive Maintenance System
emoji: ⚙️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
---

# Predictive Maintenance System

Machine Learning based engine condition prediction application.

## Overview

This application predicts engine condition using sensor readings and
engineered features generated from the input parameters.

## Input Sensors

- Engine RPM
- Lub Oil Pressure
- Fuel Pressure
- Coolant Pressure
- Lub Oil Temperature
- Coolant Temperature

## Engineered Features

- Temperature Difference
- Total Pressure
- Pressure Ratio
- RPM Temperature Interaction

## Machine Learning Model

Random Forest Classifier

## MLOps Components

- Hugging Face Dataset
- Hugging Face Model Hub
- MLflow experiment tracking
- Streamlit application
- Docker containerization
- GitHub Actions CI/CD

## Deployment

The application is packaged as a Docker-based Hugging Face Space.

The application listens on port 8501.

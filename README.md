# machine_learning_engineering_interview
## The Challenge
The aim of this problem is simple... maximize model QPS!

Currently, there are three services: loadtest (using locust), model (using fastapi and pytorch) and api (using fastapi). The API service just generates deterministic synthetic images, based on the images names it's given. The load test makes calls to the model with these images urls, letting the model fetch these images and run it through the VIT, and returning an imagenet class. 

## Getting Started
In order to get started, you will need minikube (with GPU setup), kubectl and skaffold set up. 

If you run:
```
skaffold dev --namespace dill --cleanup=false
```
skaffold will build your containers, deploy them to minikube and port-forward:
1. The model on http://localhost:8000/docs
2. The (image) api on http://localhost:8080/docs
3. Locust on http://localhost:8089


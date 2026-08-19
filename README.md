# Performance Evaluation of Deep Learning for Vehicle Detection under Adverse Environmental Conditions



This repository contains the source code, experiment notebooks, configuration files, and selected verified results produced for an MSc Artificial Intelligence research project at Sheffield Hallam University.



The project evaluates YOLO-based vehicle detection under adverse environmental conditions using both real-world datasets and controlled CARLA simulation scenarios.



## Research Scope



The study investigates the robustness and generalisation of object detection across two complementary domains:



- Real-world adverse-weather datasets

- Controlled simulation environments using CARLA



The real-world experiments use the DAWN and ACDC datasets, together with a combined dataset configuration.



The simulation experiments evaluate controlled CARLA scenarios under:



- Clear

- Rain

- Fog

- Night



The primary detection framework is based on Ultralytics YOLO.



## Repository Structure



```text

DMSc_Dissertation_GitHub/

|

|-- configs/

|   |-- ACDC/

|   |-- DAWN/

|   |-- COMBINED/

|   `-- CARLA/

|

|-- notebooks/

|   |-- ACDC_ENTIRE.ipynb

|   |-- ACDC_FOG.ipynb

|   |-- ACDC_NIGHT.ipynb

|   |-- ACDC_RAIN.ipynb

|   |-- DAWN_ENTIRE.ipynb

|   |-- DAWN_FOG.ipynb

|   |-- DAWN_RAIN.ipynb

|   |-- inference_pipeline.ipynb

|   `-- YOLO_COMBINED.ipynb

|

|-- results/

|   |-- ACDC/

|   |-- DAWN/

|   |-- COMBINED/

|   `-- CARLA/

|

|-- src/

|   |-- carla/

|   |   |-- CLEAR/

|   |   |-- RAIN/

|   |   |-- FOG/

|   |   `-- NIGHT/

|   |

|   `-- evaluation/

|       `-- utils/

|

|-- requirements.txt

|-- .gitignore

`-- README.md



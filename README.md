# Icing Detection on Outside Heat Exchanger (OHX) Using Machine Learning and Artificial Intelligence

## Project Overview

The outside heat exchanger (OHX) is an important part of an electric vehicle's heat-pump system. Under cold and humid conditions, ice can accumulate on its surface, restricting airflow and reducing heat-transfer efficiency. This can affect cabin heating performance and overall energy consumption.

My research investigated a machine-learning-based approach to identifying icing conditions in automotive heat-pump systems. The work focused on understanding the thermal-management problem and exploring whether patterns in recorded vehicle measurements could support icing detection.

## Motivation

The motivation behind this research was the need to improve the performance and reliability of automotive heat-pump systems, particularly in cold weather conditions. As electric vehicles become more widespread, efficient cabin heating is increasingly important for minimizing energy consumption and preserving driving range. However, ice formation on the outside heat exchanger can disrupt airflow and reduce heat-transfer efficiency.

Traditional icing-detection methods often rely on fixed thresholds or predefined operating points, which may not fully capture the complexity of icing under different environmental and operating conditions. Machine learning offers an alternative approach by identifying patterns and relationships in measurement data.

The objective of this research was to investigate whether machine learning could support more accurate and adaptable icing detection, with the long-term potential to improve thermal-system performance, reliability, and passenger comfort.

## Technical Fundamentals

<p align="center">
  <img src="https://github.com/user-attachments/assets/74d0284c-7723-4141-8297-523e28cfb6d8" alt="Heat-pump refrigeration cycle in heating mode" width="520">
</p>

A heat pump is an energy-efficient thermal-management system used in electric vehicles to heat the cabin by transferring heat from the outside air. In heating mode, it operates through four main components: the **compressor, condenser, expansion valve, and outside heat exchanger (OHX)**.

The refrigerant absorbs heat from the ambient air through the OHX, is compressed to increase its pressure and temperature, and releases heat inside the cabin through the condenser. The cycle then repeats. This process helps reduce electrical energy consumption and preserve driving range. However, ice formation on the OHX during cold and humid conditions can reduce heat-transfer efficiency and heating performance.

## Research Work

<p align="center">
<img width="481" height="112" alt="Data Processing Workflow" src="https://github.com/user-attachments/assets/81f09aaf-897f-48d7-b5eb-2e36cf5f505d" />
</p>

The project involved analyzing automotive measurement data, developing a proof-of-concept classification approach using Python and machine learning, and evaluating the approach within the available research conditions.

The study demonstrated the potential of data-driven methods for thermal-system diagnostics. Further validation across different environmental and vehicle operating conditions would be required before any production application.

## Modelling Workflow

<p align="center">
  <img src="https://github.com/user-attachments/assets/6b0b871e-fbf5-4221-819b-de6cdbc0162d" alt="High-level machine-learning modelling workflow" width="520">
</p>

### Modelling Approach

The research followed a supervised machine-learning workflow:

1. **Data preparation:** Recorded vehicle measurements were processed in Python and relevant signals were aligned for analysis.
2. **Labelling:** Measurement samples were grouped into iced and non-iced conditions.
3. **Model training:** An XGBoost classifier was used to learn patterns distinguishing the two classes.
4. **Evaluation:** Model predictions were assessed against labelled measurements, with further validation needed across different operating conditions.

<p align="center">
<img width="485" height="173" alt="Measured Variables" src="https://github.com/user-attachments/assets/2279a5be-cb2c-48ee-b682-dd32e2a3549a" />
</p>




### Measurement Data Processing

<p align="center">
<img width="497" height="170" alt="Code Libraries" src="https://github.com/user-attachments/assets/c1642b48-a322-494c-b282-f422b860c0af" />
</p>


Vehicle measurement data was processed using Python and the `asammdf` library. The workflow involved reading MF4 measurement files, resampling the data, and preparing labelled datasets for machine-learning analysis.

**Example: Reading and resampling an MF4 file**

```python
from asammdf import MDF

def read_mf4(file_path, sample_rate):
    mdf = MDF(file_path)
    mdf = mdf.resample(sample_rate)
    df = mdf.to_dataframe()
    return df
```

**Example: Adding a classification label**

```python
def add_condition_label(df, label):
    labelled_data = df.copy()
    labelled_data["condition_label"] = label
    return labelled_data
```

For binary classification, labels represent two conditions: icing (1) and non-icing (0).



## Comparison with Conventional Detection

| Rule-based approach | Machine-learning approach |
| --- | --- |
| Uses predefined conditions and thresholds. | Learns icing-related patterns from measurement data. |
| Often needs manual adjustment for new conditions. | Can be updated using additional labelled data. |
| Detection depends on satisfying fixed rules. | May support more flexible detection, subject to further validation. |

The intended benefit is to support more timely icing assessment and better-informed defrost decisions. Energy savings and improved performance across all operating conditions remain **potential benefits**, not demonstrated production results.

## Future Work

The thesis identifies broader testing across temperatures and operating conditions, improved robustness during changing vehicle states, and potential real-time integration into automotive control systems as further development steps.

## Research Outcome

The work contributed to research associated with a **published Mercedes-Benz patent application**. This repository presents a high-level overview of my master's research rather than its technical implementation.

## Confidentiality

The original thesis software, vehicle measurement files, training data, trained models, company-specific system details, and implementation materials are **not included** in this repository.

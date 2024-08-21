```mermaid
graph TD;
    A[methylkit.R] --> B[getting_beta_values.py];
    B --> C[annotat_script.R];
    C --> D[merging_script.py];
    D --> E[script_ageing_cpgs.py]

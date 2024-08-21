```mermaid
graph TD;
    A[Start] --> B{Is it a weekday?};
    B -->|Yes| C[Work];
    B -->|No| D[Relax];
    C --> E[End];
    D --> E[End];


The repo is to support the following functions in the future:

1. adding new functionalities like cross-validation
2. support for different interpolators,

```
bond_yield_surface_construction/
│
├── bond_yield/
│   ├── __init__.py
│   ├── interpolators/
│   │   ├── __init__.py
│   │   ├── thin_plate_spline.py
│   │   └── [other interpolators].py
│   │
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── loader.py  # For loading historical bond yield tables
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── cross_validation.py  # For performance analysis
│   │
│   └── utils/
│       ├── __init__.py
│       └── [utilities].py
│
├── tests/
│   ├── __init__.py
│   └── test_interpolators.py
│
├── setup.py  # For package installation
├── README.md
└── requirements.txt
```

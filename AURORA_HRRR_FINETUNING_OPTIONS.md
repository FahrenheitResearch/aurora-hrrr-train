# Aurora Fine-Tuning for HRRR Convective Modeling: Implementation Options

**A comprehensive guide to fine-tuning Aurora on HRRR data for high-resolution convective weather prediction, including reflectivity and supercell/tornado parameters**

---

## Overview

This document outlines various approaches to fine-tune Microsoft's Aurora weather prediction model on HRRR (High-Resolution Rapid Refresh) data to enable convective-scale modeling capabilities including radar reflectivity, storm-relative helicity, CAPE/CIN, and supercell/tornado parameters.

## Option 1: Basic Variable Extension (Recommended Starting Point)

### Approach
Extend Aurora's existing variables by adding HRRR-specific convective parameters as new surface variables.

### Implementation
```python
from aurora import AuroraPretrained
from aurora.normalisation import locations, scales

# Extended Aurora with HRRR convective variables
model = AuroraPretrained(
    surf_vars=("2t", "10u", "10v", "msl", "cape", "cin", "refc", "hlcy", "mxuphl"),
    static_vars=("lsm", "z", "slt"),
    atmos_vars=("z", "u", "v", "t", "q"),
    stabilise_level_agg=True  # Important for new variables
)
model.load_checkpoint(strict=False)

# Set normalisation statistics for new variables
locations["cape"] = 1000.0     # J/kg - typical CAPE values
locations["cin"] = -50.0       # J/kg - negative values
locations["refc"] = 20.0       # dBZ - typical reflectivity
locations["hlcy"] = 100.0      # m²/s² - storm-relative helicity
locations["mxuphl"] = 50.0     # m²/s² - updraft helicity

scales["cape"] = 1500.0
scales["cin"] = 100.0
scales["refc"] = 30.0
scales["hlcy"] = 200.0
scales["mxuphl"] = 100.0
```

### Advantages
- Minimal model architecture changes
- Leverages Aurora's existing pressure-level understanding
- Quick implementation and testing

### Disadvantages
- Treats 3D phenomena (reflectivity) as surface variables
- May not capture vertical structure of convective processes

---

## Option 2: Multi-Level Convective Variables

### Approach
Add convective variables as atmospheric variables at specific height levels rather than pressure levels.

### Implementation
```python
model = AuroraPretrained(
    surf_vars=("2t", "10u", "10v", "msl", "cape", "cin"),
    static_vars=("lsm", "z", "slt"),
    atmos_vars=("z", "u", "v", "t", "q", "refl", "w"),  # Add reflectivity and vertical velocity
    level_condition=(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
    stabilise_level_agg=True
)

# Custom data preprocessing for height-based convective variables
def convert_hrrr_to_aurora_levels(hrrr_data):
    """Convert HRRR height-based data to Aurora pressure levels"""
    # Map reflectivity from height levels to pressure levels
    # Use hydrostatic approximation for height-to-pressure conversion
    pass
```

### Advantages
- Better representation of 3D convective structure
- Can model reflectivity at multiple atmospheric levels
- More physically consistent for convective processes

### Disadvantages
- Complex height-to-pressure level mapping
- Requires custom preprocessing
- Higher computational cost

---

## Option 3: Separate Convective Head Architecture

### Approach
Add a specialized decoder head for convective variables while keeping standard meteorological variables in the main model.

### Implementation
```python
class AuroraConvective(AuroraPretrained):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Add separate decoder for convective variables
        self.convective_head = ConvectiveDecoder(
            embed_dim=512,
            output_vars=["refc", "cape", "cin", "hlcy", "mxuphl", "ustm", "vstm"]
        )
        
    def forward(self, batch):
        # Standard Aurora forward pass
        standard_pred = super().forward(batch)
        
        # Extract features from encoder for convective prediction
        features = self.encoder_features  # Custom hook needed
        convective_pred = self.convective_head(features)
        
        # Combine predictions
        return self.merge_predictions(standard_pred, convective_pred)

# Usage
model = AuroraConvective(
    modulation_head=True,  # Enable additional head capability
    separate_perceiver=("refc", "cape", "cin")
)
```

### Advantages
- Specialized processing for convective phenomena
- Can use different loss functions for different variable types
- Maintains standard meteorological prediction quality

### Disadvantages
- Complex architecture modifications
- Requires careful design of convective head
- More difficult to implement and debug

---

## Option 4: Multi-Scale Training Strategy

### Approach
Train Aurora on multiple spatial scales simultaneously, using HRRR's 3km resolution for convective features and coarser resolution for synoptic patterns.

### Implementation
```python
# Multi-scale training configuration
model = AuroraPretrained(
    surf_vars=("2t", "10u", "10v", "msl", "cape", "cin", "refc"),
    window_size=(2, 8, 16),  # Larger windows for high-res data
    patch_size=2,            # Smaller patches for 3km resolution
    encoder_depths=(8, 12, 10),  # Deeper for complex convective patterns
    stabilise_level_agg=True
)

# Custom data loader for multi-scale training
class HRRRMultiScaleDataset:
    def __init__(self):
        self.scales = [3, 6, 12]  # km resolutions
        
    def get_batch(self, scale_km):
        """Return batch at specified resolution"""
        if scale_km == 3:
            # Full HRRR resolution with convective variables
            return self.load_hrrr_native()
        else:
            # Downsampled for synoptic patterns
            return self.load_hrrr_downsampled(scale_km)
```

### Advantages
- Handles multiple scales relevant to convective prediction
- Can capture both mesoscale and convective-scale phenomena
- Flexible training strategy

### Disadvantages
- Complex data pipeline
- Increased training time and computational requirements
- Difficult to balance different scales during training

---

## Option 5: Physics-Informed Fine-Tuning

### Approach
Incorporate physical constraints and relationships between convective variables during fine-tuning.

### Implementation
```python
class PhysicsInformedLoss:
    def __init__(self):
        self.mse_loss = nn.MSELoss()
        
    def __call__(self, pred, target):
        # Standard MSE loss
        mse = self.mse_loss(pred, target)
        
        # Physics constraints
        physics_loss = 0.0
        
        # 1. CAPE-CIN relationship
        cape_pred = pred["cape"]
        cin_pred = pred["cin"]
        # High CAPE with low CIN should correlate with convective activity
        physics_loss += self.cape_cin_constraint(cape_pred, cin_pred)
        
        # 2. Reflectivity-vertical velocity relationship
        refc_pred = pred["refc"] 
        w_pred = pred.get("w", None)
        if w_pred is not None:
            physics_loss += self.reflectivity_updraft_constraint(refc_pred, w_pred)
        
        # 3. Storm motion consistency
        ustm_pred = pred.get("ustm", None)
        vstm_pred = pred.get("vstm", None)
        if ustm_pred is not None and vstm_pred is not None:
            physics_loss += self.storm_motion_constraint(ustm_pred, vstm_pred)
        
        return mse + 0.1 * physics_loss

# Training with physics constraints
model = AuroraPretrained(
    surf_vars=("2t", "10u", "10v", "msl", "cape", "cin", "refc", "ustm", "vstm"),
    positive_surf_vars=("cape", "refc"),  # Ensure physical positivity
    stabilise_level_agg=True
)

criterion = PhysicsInformedLoss()
```

### Advantages
- Incorporates meteorological knowledge
- Helps with physically consistent predictions
- Can improve generalization to extreme events

### Disadvantages
- Requires deep meteorological expertise
- Complex loss function design
- May be overly constraining for some scenarios

---

## Option 6: Ensemble-Based Convective Prediction

### Approach
Create an ensemble of specialized Aurora models, each optimized for different aspects of convective prediction.

### Implementation
```python
class ConvectiveEnsemble:
    def __init__(self):
        # Specialized models for different phenomena
        self.reflectivity_model = AuroraPretrained(
            surf_vars=("2t", "10u", "10v", "msl", "refc", "refd"),
            stabilise_level_agg=True
        )
        
        self.thermodynamic_model = AuroraPretrained(
            surf_vars=("2t", "10u", "10v", "msl", "cape", "cin", "lcl", "lfc"),
            stabilise_level_agg=True
        )
        
        self.kinematic_model = AuroraPretrained(
            surf_vars=("2t", "10u", "10v", "msl", "hlcy", "mxuphl", "ustm", "vstm"),
            stabilise_level_agg=True
        )
        
    def predict(self, batch):
        """Ensemble prediction combining all models"""
        refl_pred = self.reflectivity_model(batch)
        thermo_pred = self.thermodynamic_model(batch)
        kine_pred = self.kinematic_model(batch)
        
        return self.combine_predictions(refl_pred, thermo_pred, kine_pred)
```

### Advantages
- Specialized expertise for different convective aspects
- Can achieve high performance on specific phenomena
- Flexible ensemble weighting strategies

### Disadvantages
- Multiple models to train and maintain
- Higher computational cost for inference
- Complex ensemble combination logic

---

## Data Preparation Strategies

### Strategy 1: Direct HRRR Mapping
```python
def prepare_hrrr_batch(sfc_file, prs_file, previous_sfc_file, previous_prs_file):
    """Create Aurora batch from HRRR files"""
    
    # Current timestep data
    current_vars = extract_hrrr_variables(sfc_file, prs_file)
    
    # Previous timestep data (6 hours earlier)
    previous_vars = extract_hrrr_variables(previous_sfc_file, previous_prs_file)
    
    # Create batch with history dimension
    batch = Batch(
        surf_vars={
            "2t": torch.stack([previous_vars["t2m"], current_vars["t2m"]], dim=1),
            "10u": torch.stack([previous_vars["u10"], current_vars["u10"]], dim=1),
            "10v": torch.stack([previous_vars["v10"], current_vars["v10"]], dim=1),
            "msl": torch.stack([previous_vars["msl"], current_vars["msl"]], dim=1),
            "cape": torch.stack([previous_vars["cape"], current_vars["cape"]], dim=1),
            "refc": torch.stack([previous_vars["refc"], current_vars["refc"]], dim=1),
        },
        static_vars={
            "lsm": current_vars["lsm"],
            "z": current_vars["z_surface"],
            "slt": current_vars["slt"],
        },
        atmos_vars={
            "t": torch.stack([previous_vars["t_atmos"], current_vars["t_atmos"]], dim=1),
            "u": torch.stack([previous_vars["u_atmos"], current_vars["u_atmos"]], dim=1),
            "v": torch.stack([previous_vars["v_atmos"], current_vars["v_atmos"]], dim=1),
            "q": torch.stack([previous_vars["q_atmos"], current_vars["q_atmos"]], dim=1),
            "z": torch.stack([previous_vars["z_atmos"], current_vars["z_atmos"]], dim=1),
        },
        metadata=create_hrrr_metadata(current_vars)
    )
    
    return batch
```

### Strategy 2: Derived Variable Computation
```python
def compute_convective_parameters(batch):
    """Compute additional convective parameters from basic variables"""
    
    # Storm-relative helicity from winds and storm motion
    u10 = batch.surf_vars["10u"]
    v10 = batch.surf_vars["10v"]
    ustm = batch.surf_vars["ustm"] 
    vstm = batch.surf_vars["vstm"]
    
    # Bulk Richardson number
    brn = compute_bulk_richardson(batch.atmos_vars)
    
    # Supercell composite parameter
    scp = compute_supercell_composite(batch)
    
    # Add derived variables to batch
    batch.surf_vars["brn"] = brn
    batch.surf_vars["scp"] = scp
    
    return batch
```

---

## Training Recommendations

### Phase 1: Basic Extension (Weeks 1-2)
1. Start with Option 1 (Basic Variable Extension)
2. Add 3-5 key convective variables: CAPE, CIN, reflectivity
3. Use existing Aurora normalization approach
4. Train for 10-20 epochs with learning rate 3e-4

### Phase 2: Architecture Refinement (Weeks 3-4)
1. Implement Option 2 or 3 based on Phase 1 results
2. Add more sophisticated convective variables
3. Tune normalization statistics based on HRRR data distribution
4. Implement custom loss functions

### Phase 3: Physics Integration (Weeks 5-6)
1. Add physics-informed constraints (Option 5)
2. Validate against real convective events
3. Fine-tune for specific phenomena (supercells, squall lines)

### Phase 4: Optimization (Weeks 7-8)
1. Ensemble methods if needed (Option 6)
2. Multi-scale training optimization (Option 4)
3. Hyperparameter tuning and validation

---

## Hardware Requirements

### Minimum Configuration
- GPU: A100 80GB (for gradient computation)
- Memory: 256GB+ RAM for HRRR data processing
- Storage: 10TB+ for HRRR dataset (1 year ≈ 50TB)

### Recommended Configuration
- GPU: 4x A100 80GB for parallel training
- Memory: 512GB+ RAM
- Storage: 100TB+ with high-speed I/O
- Network: High-bandwidth for HRRR data downloading

---

## Evaluation Metrics

### Traditional Metrics
- MSE/RMSE for continuous variables
- Critical Success Index (CSI) for reflectivity thresholds
- Equitable Threat Score (ETS) for convective events

### Convective-Specific Metrics
- Probability of Detection (POD) for supercells
- False Alarm Rate (FAR) for tornado environments
- Reliability diagrams for probabilistic forecasts
- Structure-amplitude-location (SAL) metrics for reflectivity patterns

### Implementation
```python
def evaluate_convective_prediction(pred, target):
    """Comprehensive evaluation of convective predictions"""
    
    metrics = {}
    
    # Reflectivity skill
    for threshold in [20, 35, 50]:  # dBZ
        metrics[f"refc_csi_{threshold}"] = compute_csi(
            pred["refc"], target["refc"], threshold
        )
    
    # CAPE accuracy
    metrics["cape_rmse"] = torch.sqrt(torch.mean((pred["cape"] - target["cape"])**2))
    
    # Storm motion error
    storm_error = torch.sqrt(
        (pred["ustm"] - target["ustm"])**2 + 
        (pred["vstm"] - target["vstm"])**2
    )
    metrics["storm_motion_rmse"] = torch.mean(storm_error)
    
    return metrics
```

---

## Expected Outcomes

### Short-term (1-2 months)
- Basic convective parameter prediction (CAPE, CIN, reflectivity)
- Improved high-resolution precipitation forecasts
- Better representation of diurnal convective cycles

### Medium-term (3-6 months)
- Accurate storm-scale feature prediction
- Supercell and tornado environment identification
- Multi-hour convective evolution forecasting

### Long-term (6-12 months)
- Operational-quality convective forecasts at 3km resolution
- Reliable severe weather probability guidance
- Integration with ensemble forecasting systems

---

## Risk Mitigation

### Technical Risks
- **Gradient explosion**: Use `stabilise_level_agg=True` and gradient clipping
- **Memory issues**: Implement gradient checkpointing and batch size optimization
- **Overfitting**: Use dropout, early stopping, and validation on diverse events

### Data Risks
- **Quality control**: Implement comprehensive HRRR data validation
- **Missing data**: Develop interpolation strategies for gaps
- **Bias correction**: Account for HRRR model biases in training data

### Scientific Risks
- **Physical consistency**: Validate against observations and meteorological principles
- **Extreme events**: Ensure model performs on rare but important cases
- **Generalization**: Test on different seasons and geographic regions

---

This comprehensive approach provides multiple pathways to achieve high-quality convective modeling with Aurora fine-tuned on HRRR data. Start with the basic extension approach and progressively add complexity based on results and requirements.
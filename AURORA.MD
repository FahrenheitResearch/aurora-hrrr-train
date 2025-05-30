BELOW IS A PASTE OF THE VARIOUS MD FILES OF THE AURORA PAGE
# Fine-Tuning

Generally, if you wish to fine-tune Aurora for a specific application,
you should build on the pretrained version:

```python
from aurora import AuroraPretrained

model = AuroraPretrained()
model.load_checkpoint()
```

## Computing Gradients

To compute gradients, you will need an A100 with 80 GB of memory.
In addition, you will need to use [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)
and gradient checkpointing.
You can do this as follows:

```python
from aurora import AuroraPretrained

model = AuroraPretrained(autocast=True)  # Use AMP.
model.load_checkpoint()

batch = ...  # Load some data.

model = model.cuda()
model.train()
model.configure_activation_checkpointing()

pred = model.forward(batch)
loss = ...
loss.backward()
```

## Exploding Gradients

When fine-tuning, you may run into very large gradient values.
Gradient clipping and internal layer normalisation layers mitigate the impact
of large gradients,
meaning that large gradients will not immediately lead to abnormal model outputs and loss values.
Nevertheless, if gradients do blow up, the model will not learn anymore and eventually the loss value
will also blow up.
You should carefully monitor the value of the gradients to detect exploding gradients.

One cause of exploding gradients is too large values for internal activations.
Typically this can be fixed by judiciously inserting a layer normalisation layer.

We have identified the level aggregation as weak point of the model that can be susceptible
to exploding gradients.
You can stabilise the level aggregation of the model
by setting the following flag in the constructor: `stabilise_level_agg=True`.
Note that `stabilise_level_agg=True` will considerably perturb the model,
so significant additional fine-tuning may be required to get to the desired level of performance.

```python
from aurora import AuroraPretrained
from aurora.normalisation import locations, scales

model = AuroraPretrained(stabilise_level_agg=True)  # Insert extra layer norm. to mitigate exploding gradients.
model.load_checkpoint(strict=False)
```

## Extending Aurora with New Variables

Aurora can be extended with new variables by adjusting the keyword arguments `surf_vars`,
`static_vars`, and `atmos_vars`.
When you add a new variable, you also need to set the normalisation statistics.

```python
from aurora import AuroraPretrained
from aurora.normalisation import locations, scales

model = AuroraPretrained(
    surf_vars=("2t", "10u", "10v", "msl", "new_surf_var"),
    static_vars=("lsm", "z", "slt", "new_static_var"),
    atmos_vars=("z", "u", "v", "t", "q", "new_atmos_var"),
)
model.load_checkpoint(strict=False)

# Normalisation means:
locations["new_surf_var"] = 0.0
locations["new_static_var"] = 0.0
locations["new_atmos_var"] = 0.0

# Normalisation standard deviations:
scales["new_surf_var"] = 1.0
scales["new_static_var"] = 1.0
scales["new_atmos_var"] = 1.0
```

To more efficiently learn new variables, it is recommended to use a separate learning rate for
the patch embeddings of the new variables in the encoder and decoder.
For example, if you are using Adam, you can try `1e-3` for the new patch embeddings
and `3e-4` for the other parameters.

By default, patch embeddings in the encoder for new variables are initialised randomly.
This means that adding new variables to the model perturbs the predictions for the existing
variables.
If you do not want this, you can alternatively initialise the new patch embeddings in the encoder
to zero.
The relevant parameter dictionaries are `model.encoder.{surf,atmos}_token_embeds.weights`.

## Other Model Extensions

It is possible to extend to model in any way you like.
If you do this, you will likely add or remove parameters.
Then `model.load_checkpoint` will error,
because the existing checkpoint now mismatches with the model's parameters.
Simply set `model.load_checkpoint(..., strict=False)` to ignore the mismatches:

```python
from aurora import AuroraPretrained

model = AuroraPretrained(...)

... # Modify `model`.

model.load_checkpoint(strict=False)
```

## Triple Check Your Fine-Tuning Data!

When fine-tuning the model, it is absolutely essential to carefully check your fine-tuning data.

* Are the old (and possibly new) normalisation statistics appropriate for the new data?

* Is any data missing?

* Does the data contains zeros or NaNs?

* Does the data contain any outliers that could possibly interfere with fine-tuning?

_Et cetera._

# Available Models

Weights for models are made available through our [HuggingFace repository `microsoft/aurora`](https://huggingface.co/microsoft/aurora).
We now describe the available models in turn.

## Aurora 0.25° Pretrained

Aurora 0.25° Pretrained is a version of Aurora trained on a wide variety of data.

### Usage

```python
from aurora import AuroraPretrained

model = AuroraPretrained()
model.load_checkpoint()
```

### Recommended Use

Use this version of Aurora if no fine-tuned version exists for your specific data set.
For example, if you wish to make predictions for ERA5 at 0.25° resolution, this version is appropriate.
Note that 0.25° resolution means that the data has dimensions `(721, 1440)`.

Also use Aurora 0.25° Pretrained if you plan to fine-tune Aurora for you specific application,
_even if your application operates at another resolution_.

For optimal performance, the model requires the following variables and pressure levels:

| Name | Required |
| - | - |
| Surface-level variables | `2t`, `10u`, `10v`, `msl` |
| Static variables | `lsm`, `slt`, `z` |
| Atmospheric variables | `t`, `u`, `v`, `q`, `z` |
| Pressure levels (hPa) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |


### Static Variables

Aurora 0.25° Pretrained requires
[static variables from ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form).
For convenience, these are also available in
[the HuggingFace repository](https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-static.pickle).

## Aurora 0.25° Small Pretrained

Aurora 0.25° Small Pretrained is, as the name suggests, a smaller version of Aurora 0.25° Pretrained.

### Usage

```python
from aurora import AuroraSmallPretrained

model = AuroraSmallPretrained()
model.load_checkpoint()
```

### Recommended Use

Use this model for debugging purposes.
We do not recommend any other use.

## Aurora 0.25° Fine-Tuned

Aurora 0.25° Fine-Tuned is Aurora 0.25° Pretrained fine-tuned on IFS HRES T0.

### Usage

```python
from aurora import Aurora

model = Aurora()
model.load_checkpoint()
```

### Recommended Use

Use Aurora 0.25° Fine-Tuned if you aim to make predictions for IFS HRES T0.
Aurora 0.25° Fine-Tuned is the best performing version of Aurora at 0.25° resolution.

**Important:**
For optimal performance, it is crucial that you only use Aurora 0.25° Fine-Tuned for IFS HRES T0.
Producing predictions for any other data set will likely give sensible predictions,
but performance may not be optimal anymore.
[Note also that IFS HRES T0 is _not_ the same as IFS HRES analysis.](t0-vs-analysis)

For optimal performance, the model requires the following variables and pressure levels:

| Name | Required |
| - | - |
| Surface-level variables | `2t`, `10u`, `10v`, `msl` |
| Static variables | `lsm`, `slt`, `z` |
| Atmospheric variables | `t`, `u`, `v`, `q`, `z` |
| Pressure levels (hPa) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |


### Static Variables

Aurora 0.25° Fine-Tuned requires
[static variables from ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form).
For convenience, these are also available in
[the HuggingFace repository](https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-static.pickle).

(lora-or-no-lora)=
### Notes

If you require more realistic predictions are the expense of slightly higher MSE at longer lead times, you can try turning off LoRA.

| Use LoRA? | Effect |
| - | - |
| Yes | Optimal long-term MSE, but slightly blurrier predictions |
| No | More realistic predictions, but slightly higher long-term MSE |

You can turn off LoRA as follows:

```python
from aurora import Aurora

model = Aurora(use_lora=False)  # Disable LoRA for more realistic samples.
model.load_checkpoint(strict=False)
```

## Aurora 0.25° 12-Hour Pretrained

Aurora 0.25° 12-Hour Pretrained is Aurora 0.25° Pretrained with a 12-hour lead time.

### Usage

```python
from aurora import Aurora12hPretrained

model = Aurora12hPretrained()
model.load_checkpoint()
```

### Recommended Use

Use Aurora 0.25° 12-Hour Pretrained if you wish to make predictions with a 12-hour lead time.

For optimal performance, the model requires the following variables and pressure levels:

| Name | Required |
| - | - |
| Surface-level variables | `2t`, `10u`, `10v`, `msl` |
| Static variables | `lsm`, `slt`, `z` |
| Atmospheric variables | `t`, `u`, `v`, `q`, `z` |
| Pressure levels (hPa) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |


### Static Variables

Aurora 0.25° 12-Hour Pretrained requires
[static variables from ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form).
For convenience, these are also available in
[the HuggingFace repository](https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-static.pickle).


## Aurora 0.1° Fine-Tuned

Aurora 0.1° Fine-Tuned is a high-resolution version of Aurora.

### Usage

```python
from aurora import AuroraHighRes

model = AuroraHighRes()
model.load_checkpoint()
```

### Recommended Use

Use Aurora 0.1° Fine-Tuned if you aim to make predictions for IFS HRES analysis at 0.1° resolution.
Note that 0.1° resolution means that the data should have dimensions `(1801, 3600)`.
Aurora 0.1° Fine-Tuned is the best performing version of Aurora at 0.1° resolution.

**Important:**
For optimal performance, it is crucial that you only use Aurora 0.1° Fine-Tuned for IFS HRES analysis.
Producing predictions for any other data set will likely give sensible predictions,
but performance may be significantly affected.
[Note also that IFS HRES T0 is _not_ the same as IFS HRES analysis.](t0-vs-analysis)

For optimal performance, the model requires the following variables and pressure levels:

| Name | Required |
| - | - |
| Surface-level variables | `2t`, `10u`, `10v`, `msl` |
| Static variables | `lsm`, `slt`, `z` |
| Atmospheric variables | `t`, `u`, `v`, `q`, `z` |
| Pressure levels (hPa) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |


### Static Variables


Due to differences between implementations of regridding methods, it is recommended to use
[the exact static variables which we used during training](https://huggingface.co/microsoft/aurora/blob/main/aurora-0.1-static.pickle).

It is also possible to use the
[static variables from IFS HRES analysis](https://rda.ucar.edu/datasets/ds113.1/) regridded
to 0.1° resolution.
However, these static variables will not be exactly equal to the ones we used, which might impact
performance.
If you download the static variables yourself, you must adjust the normalisation statistics.
You can do that in the following way:

```python
from aurora import AuroraHighRes

model = AuroraHighRes(
    # Use manually downloaded and regridded static variables.
    surf_stats={"z": (-3.270407e03, 6.540335e04)},
)

model.load_checkpoint()
```

The specific values above should work reasonably.
<!-- Jupyter book complains that the below link doesn't work, but it does. -->
See [the API](api.rst#aurora.Aurora.__init__) for a description of `surf_vars`.
Generally, the first value in the tuple should be `min(static_z)`
and the second value `max(static_z) - min(static_z)`.

### Notes

[Like for Aurora 0.25° Fine-Tuned](lora-or-no-lora),
you can turn off LoRA to obtain more realistic predictions at the expensive of slightly higher long-term MSE:

```python
from aurora import AuroraHighRes

model = AuroraHighRes(use_lora=False)  # Disable LoRA for more realistic samples.
model.load_checkpoint(strict=False)
```

(aurora-air-pollution)=
## Aurora 0.4° Air Pollution

Aurora 0.4° Air Pollution is Aurora 0.25° Pretrained fine-tuned on
[CAMS analysis data](https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts).
This version of Aurora is capable of making air pollution forecasts.

### Usage

```python
from aurora import AuroraAirPollution

model = AuroraAirPollution()
model.load_checkpoint()
```

### Recommended Use

Use Aurora 0.4° Air Pollution if you aim to make predictions for CAMS analysis.
Note that 0.4° resolution means that the data should have dimensions `(451, 900)`.

**Important:**
For optimal performance, it is crucial that you only run Aurora 0.4° Air Pollution on CAMS analysis data.
Producing predictions for any other data set might give sensible predictions,
but performance may not be optimal anymore.

For optimal performance, the model requires the following variables and pressure levels:

| Name | Required |
| - | - |
| Surface-level variables | `2t`, `10u`, `10v`, `msl`, `pm1`, `pm2p5`, `pm10`, `tcco`, `tc_no`, `tcno2`, `tcso2`, `gtco3` |
| Static variables | `lsm`, `slt`, `z`, `static_ammonia`, `static_ammonia_log`, `static_co`, `static_co_log`, `static_nox`, `static_nox_log`, `static_so2`, `static_so2_log`  |
| Atmospheric variables | `t`, `u`, `v`, `q`, `z`, `co`, `no`, `no2`, `so2`, `go3` |
| Pressure levels (hPa) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |


### Static Variables

Aurora 0.4° Air Pollution requires
[static variables from the HuggingFace repository](https://huggingface.co/microsoft/aurora/resolve/main/aurora-0.4-air-pollution-static.pickle).

(aurora-wave)=
## Aurora 0.25° Wave

Aurora 0.25° Wave is Aurora 0.25° Pretrained fine-tuned on
[HRES-WAM ocean wave data](https://www.ecmwf.int/en/forecasts/datasets/set-ii).
This version of Aurora is capable of making ocean wave forecasts.

### Usage

```python
from aurora import AuroraWave

model = AuroraWave()
model.load_checkpoint()
```

### Recommended Use

Use Aurora 0.25° Wave if you aim to make predictions for HRES-WAM analysis data combined with HRES T0.

**Important:**
Some specific postprocessing applies to the HRES-WAM data.
See Section C.5 of the [Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-09005-y/MediaObjects/41586_2025_9005_MOESM1_ESM.pdf).

**Important:**
For optimal performance, it is crucial that you only run Aurora 0.25° Wave on batches with all
meteorological variables taken from HRES T0 and all ocean wave variables taken from HRES-WAM
analysis.
Producing predictions for any other combination might give sensible predictions,
but performance may not be optimal anymore.

For optimal performance, the model requires the following variables and pressure levels:

| Name | Required |
| - | - |
| Surface-level variables | `2t`, `10u`, `10v`, `swh`, `mwd`, `mwp`, `pp1d`, `shww`, `mdww`, `mpww`, `shts`, `mdts`, `mpts`, `swh1`, `mwd1`, `mwp1`, `swh2`, `mwd2`, `mwp2`, `10u_wave`, `10v_wave`, `wind` |
| Static variables | `lsm`, `slt`, `z`, `wmb`, `lat_mask` |
| Atmospheric variables | `t`, `u`, `v`, `q`, `z` |
| Pressure levels (hPa) | 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 |


### Static Variables

Aurora 0.25° Wave requires
[static variables from the HuggingFace repository](https://huggingface.co/microsoft/aurora/resolve/main/aurora-0.25-wave-static.pickle).


# Tropical Cyclone Tracking

Aurora has the ability to track tropical cyclones (TCs).
For tracking TCs, we recommend to use Aurora 0.25° Fine-Tuned.
The tracker is available as `aurora.Tracker`.
It should be used in conjunction with `aurora.rollout`.

Here is an example:

```python
from datetime import datetime

from aurora import Aurora, Batch, Tracker, rollout

model = Aurora()
model.load_checkpoint()

# Construct an initial condition for the model. The TC will be tracked using
# predictions for this initial condition.
initial_condition = Batch(...)

# Initialise the tracker with the current position and time of the TC. The time
# should match with the above initial condition.
tracker = Tracker(init_lat=..., init_lon=..., init_time=datetime(...))

model.eval()
model = model.to("cuda")

# Run the tracker for predictions up to two days (8 six-hour steps).
with torch.inference_mode():
    for pred in rollout(model, batch, steps=8):
        tracker.step(pred)

model = model.to("cpu")
```

Afterwards, the track can be conveniently summarised in a DataFrame:

```python
track = tracker.results()
```

[Here is a full example](example_tc_tracking) that runs the tracker to track
[Typhoon Nanmadol](https://en.wikipedia.org/wiki/Typhoon_Nanmadol_(2022)).


# Form of a Batch

You must feed data to the model in the form of a `aurora.Batch`.
We now explain the exact form of `aurora.Batch`.

## Overall Structure

Batches contain four things:

1. some surface-level variables,
2. some static variables,
3. some atmospheric variables all at the same collection of pressure levels, and
4. metadata describing these variables: latitudes, longitudes,
    the pressure levels of the atmospheric variables, and the time of the data.

All variables in a batch are unnormalised.
Normalisation happens internally in the model.

Before we explain the four components in detail, here is an example with randomly generated data:

```python
from datetime import datetime

import torch

from aurora import Batch, Metadata

batch = Batch(
    surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
    static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
    atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
    metadata=Metadata(
        lat=torch.linspace(90, -90, 17),
        lon=torch.linspace(0, 360, 32 + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)
```

## `Batch.surf_vars`

`Batch.surf_vars` is a dictionary mapping names of surface-level variables to the numerical values
of the variables.
The surface-level variables must be of the form `(b, t, h, w)` where `b` is the batch size,
`t` the history dimension, `h` the number of latitudes, and `w` the number of longitudes.

All Aurora models produce the prediction for the next step from the current _and_ previous step.
`surf_vars[:, 1, :, :]` must correspond to the current step,
and `surf_vars[:, 0, :, :]` must correspond to the previous step, so the step before that.

The following surface-level variables are allowed:

| Name | Description |
| - | - |
| `2t` | Two-meter temperature in `K` |
| `10u` | Ten-meter eastward wind speed in `m/s` |
| `10v` | Ten-meter southward wind speed in `m/s` |
| `msl` | Mean sea-level pressure in `Pa` |

For [Aurora 0.4° Air Pollution](aurora-air-pollution), the following surface-level variables are
also allowed:

| Name | Description |
| - | - |
| `pm1` | Particulate matter less than `1 um` in `kg/m^3` |
| `pm2p5` | Particulate matter less than `2.5 um` in `kg/m^3` |
| `pm10` | Particulate matter less than `10 um` in `kg/m^3` |
| `tcco` | Total column carbon monoxide in `kg/m^2` |
| `tc_no` | Total column nitrogen monoxide in `kg/m^2` |
| `tcno2` | Total column nitrogen dioxide in `kg/m^2` |
| `tcso2` | Total column sulphur dioxide in `kg/m^2` |
| `gtco3` | Total column ozone in `kg/m^2` |

For [Aurora 0.25° Wave](aurora-wave), the following surface-level variables are also allowed:

| Name | Description |
| - | - |
| `swh` | Significant wave height of the total wave in `m` |
| `mwd` | Mean wave direction of the total wave in `degrees` |
| `mwp` | Mean wave period of the total wave in `s` |
| `pp1d` | Peak wave period of the total wave in `s` |
| `shww` | Significant wave height of the wind wave component in `m` |
| `mdww` | Mean wave direction of the wind wave component in `degrees` |
| `mpww` | Mean wave period of the wind wave component in `s` |
| `shts` | Significant wave height of the total swell component in `m` |
| `mdts` | Mean wave direction of the total swell component in `degrees` |
| `mpts` | Mean wave period of the total swell component in `s` |
| `swh1` | Significant wave height of the first swell component in `m` |
| `mwd1` | Mean wave direction of the first swell component in `degrees` |
| `mwp1` | Mean wave period of the first swell component in `s` |
| `swh2` | Significant wave height of the second swell component in `m` |
| `mwd2` | Mean wave direction of the second swell component in `degrees` |
| `mwp2` | Mean wave period of the second swell component in `s` |
| `wind` | Ten-meter neutral wind speed in `m/s` |
| `10u_wind` | Ten-meter eastward neutral wind speed in `m/s` |
| `10v_wind` | Ten-meter southward neutral wind speed in `m/s` |

## `Batch.static_vars`

`Batch.static_vars` is a dictionary mapping names of static variables to the
numerical values of the variables.
The static variables must be of the form `(h, w)` where `h` is the number of latitudes
and `w` the number of longitudes.

The following static variables are allowed:

| Name | Description |
| - | - |
| `lsm` | [Land-sea mask](https://codes.ecmwf.int/grib/param-db/172) |
| `slt` | [Soil type](https://codes.ecmwf.int/grib/param-db/43) |
| `z` | Surface-level geopotential in `m^2/s^2` |

[Aurora 0.4° Air Pollution](aurora-air-pollution)
and [Aurora 0.25° Wave](aurora-wave) require additional static variables, but these are not
easy to obtain yourself.
You need to obtain these from the HuggingFace repository.
See the description of the models.

## `Batch.atmos_vars`

`Batch.atmos_vars` is a dictionary mapping names of atmospheric variables to the
numerical values of the variables.
The atmospheric variables must be of the form `(b, t, c, h, w)` where `b` is the batch size,
`t` the history dimension, `c` the number of pressure levels, `h` the number of latitudes,
and `w` the number of longitudes.
All atmospheric variables must contain the same collection of pressure levels in the same order.

The following atmospheric variables are allowed:

| Name | Description |
| - | - |
| `t` | Temperature in `K` |
| `u` | Eastward wind speed in `m/s` |
| `v` | Southward wind speed in `m/s` |
| `q` | Specific humidity in `kg/kg` |
| `z` | Geopotential in `m^2/s^2` |

For [Aurora 0.4° Air Pollution](aurora-air-pollution), the following atmospheric variables are
also allowed:

| Name | Description |
| - | - |
| `co` | Carbon monoxide in `kg/kg` |
| `no` | Nitrogen monoxide in `kg/kg` |
| `no2` | Nitrogen dioxide in `kg/kg` |
| `so2` | Sulphur dioxide in `kg/kg` |
| `go3` | Ozone in `kg/kg` |

## `Batch.metadata`

`Batch.metadata` must be a `Metadata`, which contains the following fields:

* `Metadata.lat` is the vector of latitudes.
    The latitudes must be _decreasing_.
    The latitudes can either include both endpoints, like `linspace(90, -90, 721)`,
    or not include the south pole, like `linspace(90, -90, 721)[:-1]`.
    For curvilinear grids, this can also be a matrix, in which case the foregoing conditions
    apply to every _column_.
* `Metadata.lon` is the vector of longitudes.
    The longitudes must be _increasing_.
    The longitudes must be in the range `[0, 360)`, so they can include zero and cannot include 360.
    For curvilinear grids, this can also be a matrix, in which case the foregoing conditions
    apply to every _row_.
* `Metadata.atmos_levels` is a `tuple` of the pressure levels of the atmospheric variables in hPa.
    Note that these levels must be in exactly correspond to the order of the atmospheric variables.
    Note also that `Metadata.atmos_levels` should be a `tuple`, not a `list`.
* `Metadata.time` is a `tuple` with, for each batch element, a `datetime.datetime` representing the time of the data.
    If the batch size is one, then this will be a one-element `tuple`, e.g. `(datetime(2024, 1, 1, 12, 0),)`.
    Since all Aurora models require variables for the current _and_ previous step,
    `Metadata.time` corresponds to the time of the _current_ step.
    Specifically, `Metadata.time[i]` corresponds to the time of `Batch.surf_vars[i, -1]`.

## Model Output

The output of `aurora.forward(batch)` will again be a `Batch`.
This batch is of exactly the same form, with only one difference:
the history dimension will have size one.

Application Programming Interface
Batch
class aurora.Batch(surf_vars: dict[str, Tensor], static_vars: dict[str, Tensor], atmos_vars: dict[str, Tensor], metadata: Metadata)[source]
A batch of data.

Parameters
:
surf_vars (dict[str, torch.Tensor]) – Surface-level variables with shape (b, t, h, w).

static_vars (dict[str, torch.Tensor]) – Static variables with shape (h, w).

atmos_vars (dict[str, torch.Tensor]) – Atmospheric variables with shape (b, t, c, h, w).

metadata (Metadata) – Metadata associated to this batch.

crop(patch_size: int) → Batch[source]
Crop the variables in the batch to patch size patch_size.

classmethod from_netcdf(path: str | Path) → Batch[source]
Load a batch from a file.

normalise(surf_stats: dict[str, tuple[float, float]]) → Batch[source]
Normalise all variables in the batch.

Parameters
:
surf_stats (dict[str, tuple[float, float]]) – For these surface-level variables, adjust the normalisation to the given tuple consisting of a new location and scale.

Returns
:
Normalised batch.

Return type
:
Batch

regrid(res: float) → Batch[source]
Regrid the batch to a res degrees resolution.

This results in float32 data on the CPU.

This function is not optimised for either speed or accuracy. Use at your own risk.

property spatial_shape: tuple[int, int]
Get the spatial shape from an arbitrary surface-level variable.

to(device: str | device) → Batch[source]
Move the batch to another device.

to_netcdf(path: str | Path) → None[source]
Write the batch to a file.

This requires xarray and netcdf4 to be installed.

type(t: type) → Batch[source]
Convert everything to type t.

unnormalise(surf_stats: dict[str, tuple[float, float]]) → Batch[source]
Unnormalise all variables in the batch.

Parameters
:
surf_stats (dict[str, tuple[float, float]]) – For these surface-level variables, adjust the normalisation to the given tuple consisting of a new location and scale.

Returns
:
Unnormalised batch.

Return type
:
Batch

class aurora.Metadata(lat: Tensor, lon: Tensor, time: tuple[datetime, ...], atmos_levels: tuple[int | float, ...], rollout_step: int = 0)[source]
Metadata in a batch.

Parameters
:
lat (torch.Tensor) – Latitudes.

lon (torch.Tensor) – Longitudes.

time (tuple[datetime, ...]) – For every batch element, the time.

atmos_levels (tuple[int | float, ...]) – Pressure levels for the atmospheric variables in hPa.

rollout_step (int, optional) – How many roll-out steps were used to produce this prediction. If equal to 0, which is the default, then this means that this is not a prediction, but actual data. This field is automatically populated by the model and used to use a separate LoRA for every roll-out step. Generally, you are safe to ignore this field.

Roll-Outs
class aurora.rollout(model: Aurora, batch: Batch, steps: int)[source]
Perform a roll-out to make long-term predictions.

Parameters
:
model (aurora.model.aurora.Aurora) – The model to roll out.

batch (aurora.batch.Batch) – The batch to start the roll-out from.

steps (int) – The number of roll-out steps.

Yields
:
aurora.batch.Batch – The prediction after every step.

Tropical Cyclone Tracking
class aurora.Tracker(init_lat: float, init_lon: float, init_time: datetime)[source]
Simple tropical cyclone tracker.

This algorithm was originally designed and implemented by Anna Allen. This particular implementation is by Wessel Bruinsma and features various improvements over the original design.

results() → DataFrame[source]
Assemble the track into a convenient DataFrame.

step(batch: Batch) → None[source]
Track the next step.

Parameters
:
batch (aurora.batch.Batch) – Prediction.

Models
class aurora.Aurora(*, surf_vars: tuple[str, ...] = ('2t', '10u', '10v', 'msl'), static_vars: tuple[str, ...] = ('lsm', 'z', 'slt'), atmos_vars: tuple[str, ...] = ('z', 'u', 'v', 't', 'q'), window_size: tuple[int, int, int] = (2, 6, 12), encoder_depths: tuple[int, ...] = (6, 10, 8), encoder_num_heads: tuple[int, ...] = (8, 16, 32), decoder_depths: tuple[int, ...] = (8, 10, 6), decoder_num_heads: tuple[int, ...] = (32, 16, 8), latent_levels: int = 4, patch_size: int = 4, embed_dim: int = 512, num_heads: int = 16, mlp_ratio: float = 4.0, drop_path: float = 0.0, drop_rate: float = 0.0, enc_depth: int = 1, dec_depth: int = 1, dec_mlp_ratio: float = 2.0, perceiver_ln_eps: float = 1e-05, max_history_size: int = 2, timestep: timedelta = datetime.timedelta(seconds=21600), stabilise_level_agg: bool = False, use_lora: bool = True, lora_steps: int = 40, lora_mode: Literal['single', 'from_second', 'all'] = 'single', surf_stats: dict[str, tuple[float, float]] | None = None, autocast: bool = False, level_condition: tuple[int | float, ...] | None = None, dynamic_vars: bool = False, atmos_static_vars: bool = False, separate_perceiver: tuple[str, ...] = (), modulation_head: bool = False, positive_surf_vars: tuple[str, ...] = (), positive_atmos_vars: tuple[str, ...] = (), simulate_indexing_bug: bool = False)[source]
The Aurora model.

Defaults to the 1.3 B parameter configuration.

__init__(*, surf_vars: tuple[str, ...] = ('2t', '10u', '10v', 'msl'), static_vars: tuple[str, ...] = ('lsm', 'z', 'slt'), atmos_vars: tuple[str, ...] = ('z', 'u', 'v', 't', 'q'), window_size: tuple[int, int, int] = (2, 6, 12), encoder_depths: tuple[int, ...] = (6, 10, 8), encoder_num_heads: tuple[int, ...] = (8, 16, 32), decoder_depths: tuple[int, ...] = (8, 10, 6), decoder_num_heads: tuple[int, ...] = (32, 16, 8), latent_levels: int = 4, patch_size: int = 4, embed_dim: int = 512, num_heads: int = 16, mlp_ratio: float = 4.0, drop_path: float = 0.0, drop_rate: float = 0.0, enc_depth: int = 1, dec_depth: int = 1, dec_mlp_ratio: float = 2.0, perceiver_ln_eps: float = 1e-05, max_history_size: int = 2, timestep: timedelta = datetime.timedelta(seconds=21600), stabilise_level_agg: bool = False, use_lora: bool = True, lora_steps: int = 40, lora_mode: Literal['single', 'from_second', 'all'] = 'single', surf_stats: dict[str, tuple[float, float]] | None = None, autocast: bool = False, level_condition: tuple[int | float, ...] | None = None, dynamic_vars: bool = False, atmos_static_vars: bool = False, separate_perceiver: tuple[str, ...] = (), modulation_head: bool = False, positive_surf_vars: tuple[str, ...] = (), positive_atmos_vars: tuple[str, ...] = (), simulate_indexing_bug: bool = False) → None[source]
Construct an instance of the model.

Parameters
:
surf_vars (tuple[str, ...], optional) – All surface-level variables supported by the model.

static_vars (tuple[str, ...], optional) – All static variables supported by the model.

atmos_vars (tuple[str, ...], optional) – All atmospheric variables supported by the model.

window_size (tuple[int, int, int], optional) – Vertical height, height, and width of the window of the underlying Swin transformer.

encoder_depths (tuple[int, ...], optional) – Number of blocks in each encoder layer.

encoder_num_heads (tuple[int, ...], optional) – Number of attention heads in each encoder layer. The dimensionality doubles after every layer. To keep the dimensionality of every head constant, you want to double the number of heads after every layer. The dimensionality of attention head of the first layer is determined by embed_dim divided by the value here. For all cases except one, this is equal to 64.

decoder_depths (tuple[int, ...], optional) – Number of blocks in each decoder layer. Generally, you want this to be the reversal of encoder_depths.

decoder_num_heads (tuple[int, ...], optional) – Number of attention heads in each decoder layer. Generally, you want this to be the reversal of encoder_num_heads.

latent_levels (int, optional) – Number of latent pressure levels.

patch_size (int, optional) – Patch size.

embed_dim (int, optional) – Patch embedding dimension.

num_heads (int, optional) – Number of attention heads in the aggregation and deaggregation blocks. The dimensionality of these attention heads will be equal to embed_dim divided by this value.

mlp_ratio (float, optional) – Hidden dim. to embedding dim. ratio for MLPs.

drop_rate (float, optional) – Drop-out rate.

drop_path (float, optional) – Drop-path rate.

enc_depth (int, optional) – Number of Perceiver blocks in the encoder.

dec_depth (int, optioanl) – Number of Perceiver blocks in the decoder.

dec_mlp_ratio (float, optional) – Hidden dim. to embedding dim. ratio for MLPs in the decoder. The embedding dimensionality here is different, which is why this is a separate parameter.

perceiver_ln_eps (float, optional) – Epsilon in the perceiver layer norm. layers. Used to stabilise the model.

max_history_size (int, optional) – Maximum number of history steps. You can load checkpoints with a smaller max_history_size, but you cannot load checkpoints with a larger max_history_size.

timestep (timedelta, optional) – Timestep of the model. Defaults to 6 hours.

stabilise_level_agg (bool, optional) – Stabilise the level aggregation by inserting an additional layer normalisation. Defaults to False.

use_lora (bool, optional) – Use LoRA adaptation.

lora_steps (int, optional) – Use different LoRA adaptation for the first so-many roll-out steps.

lora_mode (str, optional) – LoRA mode. “single” uses the same LoRA for all roll-out steps, “from_second” uses the same LoRA from the second roll-out step on, and “all” uses a different LoRA for every roll-out step. Defaults to “single”.

surf_stats (dict[str, tuple[float, float]], optional) – For these surface-level variables, adjust the normalisation to the given tuple consisting of a new location and scale.

autocast (bool, optional) – Use torch.autocast to reduce memory usage. Defaults to False.

level_condition (tuple[int | float, ...], optional) – Make the patch embeddings dependent on pressure level. If you want to enable this feature, provide a tuple of all possible pressure levels.

dynamic_vars (bool, optional) – Use dynamically generated static variables, like time of day. Defaults to False.

atmos_static_vars (bool, optional) – Also concatenate the static variables to the atmospheric variables. Defaults to False.

separate_perceiver (tuple[str, ...], optional) – In the decoder, use a separate Perceiver for specific atmospheric variables. This can be helpful at fine-tuning time to deal with variables that have a significantly different behaviour. If you want to enable this features, set this to the collection of variables that should be run on a separate Perceiver.

modulation_head (bool, optional) – Enable an additional head, the so-called modulation head, that can be used to predict the difference. Defaults to False.

positive_surf_vars (tuple[str, ...], optional) – Mark these surface-level variables as positive. Clamp them before running them through the encoder, and also clamp them when autoregressively rolling out the model. The variables are not clamped for the first roll-out step.

positive_atmos_vars (tuple[str, ...], optional) – Mark these atmospheric variables as positive. Clamp them before running them through the encoder, and also clamp them when autoregressively rolling out the model. The variables are not clamped for the first roll-out step.

simulate_indexing_bug (bool, optional) – Simulate an indexing bug that’s present for the air pollution version of Aurora. This is necessary to obtain numerical equivalence to the original implementation. Defaults to False.

adapt_checkpoint_max_history_size(checkpoint: dict[str, Tensor]) → None[source]
Adapt a checkpoint with smaller max_history_size to a model with a larger max_history_size than the current model.

If a checkpoint was trained with a larger max_history_size than the current model, this function will assert fail to prevent loading the checkpoint. This is to prevent loading a checkpoint which will likely cause the checkpoint to degrade is performance.

This implementation copies weights from the checkpoint to the model and fills zeros for the new history width dimension. It mutates checkpoint.

batch_transform_hook(batch: Batch) → Batch[source]
Transform the batch right after receiving it and before normalisation.

This function should be idempotent.

configure_activation_checkpointing()[source]
Configure activation checkpointing.

This is required in order to compute gradients without running out of memory.

default_checkpoint_name = 'aurora-0.25-finetuned.ckpt'
Name of the default checkpoint.

Type
:
str

default_checkpoint_repo = 'microsoft/aurora'
Name of the HuggingFace repository to load the default checkpoint from.

Type
:
str

forward(batch: Batch) → Batch[source]
Forward pass.

Parameters
:
batch (Batch) – Batch to run the model on.

Returns
:
Prediction for the batch.

Return type
:
Batch

load_checkpoint(repo: str | None = None, name: str | None = None, strict: bool = True) → None[source]
Load a checkpoint from HuggingFace.

Parameters
:
repo (str, optional) – Name of the repository of the form user/repo.

name (str, optional) – Path to the checkpoint relative to the root of the repository, e.g. checkpoint.cpkt.

strict (bool, optional) – Error if the model parameters are not exactly equal to the parameters in the checkpoint. Defaults to True.

load_checkpoint_local(path: str, strict: bool = True) → None[source]
Load a checkpoint directly from a file.

Parameters
:
path (str) – Path to the checkpoint.

strict (bool, optional) – Error if the model parameters are not exactly equal to the parameters in the checkpoint. Defaults to True.

class aurora.AuroraPretrained(*, use_lora: bool = False, **kw_args)[source]
Pretrained version of Aurora.

default_checkpoint_name = 'aurora-0.25-pretrained.ckpt'
Name of the default checkpoint.

Type
:
str

class aurora.AuroraSmallPretrained(*, encoder_depths: tuple[int, ...] = (2, 6, 2), encoder_num_heads: tuple[int, ...] = (4, 8, 16), decoder_depths: tuple[int, ...] = (2, 6, 2), decoder_num_heads: tuple[int, ...] = (16, 8, 4), embed_dim: int = 256, num_heads: int = 8, use_lora: bool = False, **kw_args)[source]
Small pretrained version of Aurora.

Should only be used for debugging.

default_checkpoint_name = 'aurora-0.25-small-pretrained.ckpt'
Name of the default checkpoint.

Type
:
str

class aurora.Aurora12hPretrained(*, timestep: timedelta = datetime.timedelta(seconds=43200), use_lora: bool = False, **kw_args)[source]
Pretrained version of Aurora with time step 12 hours.

default_checkpoint_name = 'aurora-0.25-12h-pretrained.ckpt'
Name of the default checkpoint.

Type
:
str

class aurora.AuroraHighRes(*, surf_vars: tuple[str, ...] = ('2t', '10u', '10v', 'msl'), static_vars: tuple[str, ...] = ('lsm', 'z', 'slt'), atmos_vars: tuple[str, ...] = ('z', 'u', 'v', 't', 'q'), window_size: tuple[int, int, int] = (2, 6, 12), encoder_depths: tuple[int, ...] = (6, 10, 8), encoder_num_heads: tuple[int, ...] = (8, 16, 32), decoder_depths: tuple[int, ...] = (8, 10, 6), decoder_num_heads: tuple[int, ...] = (32, 16, 8), latent_levels: int = 4, patch_size: int = 4, embed_dim: int = 512, num_heads: int = 16, mlp_ratio: float = 4.0, drop_path: float = 0.0, drop_rate: float = 0.0, enc_depth: int = 1, dec_depth: int = 1, dec_mlp_ratio: float = 2.0, perceiver_ln_eps: float = 1e-05, max_history_size: int = 2, timestep: timedelta = datetime.timedelta(seconds=21600), stabilise_level_agg: bool = False, use_lora: bool = True, lora_steps: int = 40, lora_mode: Literal['single', 'from_second', 'all'] = 'single', surf_stats: dict[str, tuple[float, float]] | None = None, autocast: bool = False, level_condition: tuple[int | float, ...] | None = None, dynamic_vars: bool = False, atmos_static_vars: bool = False, separate_perceiver: tuple[str, ...] = (), modulation_head: bool = False, positive_surf_vars: tuple[str, ...] = (), positive_atmos_vars: tuple[str, ...] = (), simulate_indexing_bug: bool = False)[source]
High-resolution version of Aurora.

default_checkpoint_name = 'aurora-0.1-finetuned.ckpt'
Name of the default checkpoint.

Type
:
str

class aurora.AuroraAirPollution(*, surf_vars: tuple[str, ...] = ('2t', '10u', '10v', 'msl', 'pm1', 'pm2p5', 'pm10', 'tcco', 'tc_no', 'tcno2', 'gtco3', 'tcso2'), static_vars: tuple[str, ...] = ('lsm', 'z', 'slt', 'static_ammonia', 'static_ammonia_log', 'static_co', 'static_co_log', 'static_nox', 'static_nox_log', 'static_so2', 'static_so2_log'), atmos_vars: tuple[str, ...] = ('z', 'u', 'v', 't', 'q', 'co', 'no', 'no2', 'go3', 'so2'), patch_size: int = 3, timestep: timedelta = datetime.timedelta(seconds=43200), level_condition: tuple[int | float, ...] | None = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000), dynamic_vars: bool = True, atmos_static_vars: bool = True, separate_perceiver: tuple[str, ...] = ('co', 'no', 'no2', 'go3', 'so2'), modulation_head: bool = True, positive_surf_vars: tuple[str, ...] = ('pm1', 'pm2p5', 'pm10', 'tcco', 'tc_no', 'tcno2', 'gtco3', 'tcso2'), positive_atmos_vars: tuple[str, ...] = ('co', 'no', 'no2', 'go3', 'so2'), simulate_indexing_bug: bool = True, **kw_args)[source]
Fine-tuned version of Aurora for air pollution.

default_checkpoint_name = 'aurora-0.4-air-pollution.ckpt'
Name of the default checkpoint.

Type
:
str

class aurora.AuroraWave(*, surf_vars: tuple[str, ...] = ('2t', '10u', '10v', 'msl', 'swh', 'mwd', 'mwp', 'pp1d', 'shww', 'mdww', 'mpww', 'shts', 'mdts', 'mpts', 'swh1', 'mwd1', 'mwp1', 'swh2', 'mwd2', 'mwp2', 'wind', '10u_wave', '10v_wave'), static_vars: tuple[str, ...] = ('lsm', 'z', 'slt', 'wmb', 'lat_mask'), lora_mode: Literal['single', 'from_second', 'all'] = 'from_second', stabilise_level_agg: bool = True, density_channel_surf_vars: tuple[str, ...] = ('swh', 'mwd', 'mwp', 'pp1d', 'shww', 'mdww', 'mpww', 'shts', 'mdts', 'mpts', 'swh1', 'mwd1', 'mwp1', 'swh2', 'mwd2', 'mwp2', 'wind', '10u_wave', '10v_wave'), angle_surf_vars: tuple[str, ...] = ('mwd', 'mdww', 'mdts', 'mwd1', 'mwd2'), **kw_args)[source]
Version of Aurora fined-tuned to HRES-WAM ocean wave data.

batch_transform_hook(batch: Batch) → Batch[source]
Transform the batch right after receiving it and before normalisation.

This function should be idempotent.

default_checkpoint_name = 'aurora-0.25-wave.ckpt'
Name of the default checkpoint.

Type
:
str

Predictions for HRES at 0.1 Degrees
In this example, we will download IFS HRES analysis data at 0.1 degrees resolution for 11 May 2022 from the Research Data Archive and run Aurora on this data. We will use the version of Aurora that was fine-tuned on IFS HRES analysis at 0.1 degrees resolution.

Running this notebook requires additional Python packages. You can install these as follows:

pip install requests cfgrib matplotlib
Downloading the Data
To start, we download the data from Research Data Archive.

from datetime import datetime
from pathlib import Path

import requests
import xarray as xr

# Data will be downloaded here.
download_path = Path("~/downloads/hres_0.1").expanduser()

# Day to download. This will download all times for that day.
date = datetime(2022, 5, 11)

# Each variable has a number associated with it. This is the number that will be used in
# the RDA request.
var_nums = {
    "2t": "167",  # 2m temperature
    "10u": "165",  # 10m u-component of wind
    "10v": "166",  # 10m v-component of wind
    "msl": "151",  # Mean sea level pressure
    "t": "130",  # Temperature
    "u": "131",  # u-component of wind (atmospheric)
    "v": "132",  # v-component of wind (atmospheric)
    "q": "133",  # Specific humidity (atmospheric)
    "z": "129",  # Geopotential
    "slt": "043",  # Soil type
    "lsm": "172",  # Land-sea mask
}

# Construct the URLs to download the data from.
downloads: dict[Path, str] = {}
for v in ["2t", "10u", "10v", "msl", "z", "slt", "lsm"]:
    downloads[download_path / date.strftime(f"surf_{v}_%Y-%m-%d.grib")] = (
        f"https://data.rda.ucar.edu/ds113.1/"
        f"ec.oper.an.sfc/{date.year}{date.month:02d}/ec.oper.an.sfc.128_{var_nums[v]}_{v}."
        f"regn1280sc.{date.year}{date.month:02d}{date.day:02d}.grb"
    )
for v in ["z", "t", "u", "v", "q"]:
    for hour in [0, 6, 12, 18]:
        prefix = "uv" if v in {"u", "v"} else "sc"
        downloads[download_path / date.strftime(f"atmos_{v}_%Y-%m-%d_{hour:02d}.grib")] = (
            f"https://data.rda.ucar.edu/ds113.1/"
            f"ec.oper.an.pl/{date.year}{date.month:02d}/ec.oper.an.pl.128_{var_nums[v]}_{v}."
            f"regn1280{prefix}.{date.year}{date.month:02d}{date.day:02d}{hour:02d}.grb"
        )

# Perform the downloads.
for target, source in downloads.items():
    if not target.exists():
        print(f"Downloading {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(source)
        response.raise_for_status()
        with open(target, "wb") as f:
            f.write(response.content)
print("Downloads finished!")
Downloads finished!
Preparing a Batch
We convert the downloaded data to an aurora.Batch, which is what the model requires.

import numpy as np
import torch

from aurora import Batch, Metadata

# Load these pressure levels.
levels = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)


def load_surf(v: str, v_in_file: str) -> torch.Tensor:
    """Load the downloaded surface-level or static variable `v` for hours 00 and 06."""
    ds = xr.open_dataset(download_path / date.strftime(f"surf_{v}_%Y-%m-%d.grib"), engine="cfgrib")
    data = ds[v_in_file].values[:2]  # Use hours 00 and 06.
    data = data[None]  # Insert a batch dimension.
    return torch.from_numpy(data)


def load_atmos(v: str) -> torch.Tensor:
    """Load the downloaded atmospheric variable `v` for hours 00 and 06."""
    ds_00 = xr.open_dataset(
        download_path / date.strftime(f"atmos_{v}_%Y-%m-%d_00.grib"), engine="cfgrib"
    )
    ds_06 = xr.open_dataset(
        download_path / date.strftime(f"atmos_{v}_%Y-%m-%d_06.grib"), engine="cfgrib"
    )
    # Select the right pressure levels.
    ds_00 = ds_00[v].sel(isobaricInhPa=list(levels))
    ds_06 = ds_06[v].sel(isobaricInhPa=list(levels))
    data = np.stack((ds_00.values, ds_06.values), axis=0)
    data = data[None]  # Insert a batch dimension.
    return torch.from_numpy(data)


# Extract the latitude and longitude from an arbitrary downloaded file.
ds = xr.open_dataset(next(iter(downloads.keys())), engine="cfgrib")

batch = Batch(
    surf_vars={
        "2t": load_surf("2t", "t2m"),
        "10u": load_surf("10u", "u10"),
        "10v": load_surf("10v", "v10"),
        "msl": load_surf("msl", "msl"),
    },
    static_vars={
        # The static variables are constant, so we just get them for the first time.
        "z": load_surf("z", "z")[0, 0],
        "slt": load_surf("slt", "slt")[0, 0],
        "lsm": load_surf("lsm", "lsm")[0, 0],
    },
    atmos_vars={
        "t": load_atmos("t"),
        "u": load_atmos("u"),
        "v": load_atmos("v"),
        "q": load_atmos("q"),
        "z": load_atmos("z"),
    },
    metadata=Metadata(
        lat=torch.from_numpy(ds.latitude.values),
        lon=torch.from_numpy(ds.longitude.values),
        time=(date.replace(hour=6),),
        atmos_levels=levels,
    ),
)
# Regrid the batch to 0.1 degrees resolution. Note that this is a convenience function
# which is not optimised for speed or accuracy. Use at your own risk!
batch = batch.regrid(res=0.1)
Loading and Running the Model
Finally, we are ready to load and run the model and visualise the predictions. We perform a roll-out for two steps, which produces predictions for hours 12:00 and 18:00.

from aurora import AuroraHighRes, rollout

model = AuroraHighRes(
    # Use manually downloaded and regridded static variables.
    surf_stats={"z": (-3.270407e03, 6.540335e04)},
)
model.load_checkpoint("microsoft/aurora", "aurora-0.1-finetuned.ckpt")

model.eval()
model = model.to("cuda")

with torch.inference_mode():
    preds = [pred.to("cpu") for pred in rollout(model, batch, steps=2)]

model = model.to("cpu")
import matplotlib.pyplot as plt

truth = xr.open_dataset(download_path / date.strftime("surf_2t_%Y-%m-%d.grib"), engine="cfgrib")

fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

for i in range(ax.shape[0]):
    pred = preds[i]

    ax[i, 0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50)
    ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
    if i == 0:
        ax[i, 0].set_title("Aurora Prediction")
    ax[i, 0].set_xticks([])
    ax[i, 0].set_yticks([])

    ref = truth["t2m"][2 + i].values
    ax[i, 1].imshow(ref - 273.15, vmin=-50, vmax=50)
    if i == 0:
        ax[i, 1].set_title("HRES Analysis")
    ax[i, 1].set_xticks([])
    ax[i, 1].set_yticks([])

plt.tight_layout()



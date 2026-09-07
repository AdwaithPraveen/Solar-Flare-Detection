# SWANSF Dataset — Attribute Descriptions

The SWANSF dataset contains multivariate time-series observations of solar active regions. Each sample contains multiple timestamps and 24 solar-physics attributes.

## Attributes

### 1. `R_VALUE`

**Full meaning:** Schrijver's R Parameter

Measures the amount of magnetic flux near strong, high-gradient polarity inversion lines (MPILs) in a solar active region.

* Indicates magnetic complexity near polarity inversion lines.
* Higher values generally indicate stronger and more compact magnetic gradients.
* Useful for estimating solar flare potential.

**Unit:** Mx (Maxwell)

---

### 2. `TOTUSJH`

**Full meaning:** Total Unsigned Current Helicity

Measures the total unsigned current helicity of the active region.

* Indicates magnetic twist and electric-current structures.
* Higher values indicate stronger magnetic non-potentiality.
* Useful for studying solar-flare-related magnetic complexity.

**Unit:** G²/m

---

### 3. `TOTBSQ`

**Full meaning:** Total Magnetic Field Squared

Represents the sum/integral of the squared magnetic-field strength over the active region.

* Measures the overall strength of the magnetic field.
* Higher values indicate stronger magnetic fields.
* Related to the magnetic energy stored in the active region.

**Unit:** G²

---

### 4. `TOTPOT`

**Full meaning:** Total Magnetic Free Energy Density

Measures the magnetic free-energy-related quantity of the active region.

* Represents energy associated with the non-potential component of the magnetic field.
* Higher values indicate greater stored magnetic energy.
* Important for solar flares and coronal mass ejections.

**Unit:** erg/cm³

---

### 5. `TOTUSJZ`

**Full meaning:** Total Unsigned Vertical Current

Measures the total unsigned vertical electric current in the active region.

* Indicates the overall strength of electric-current structures.
* Higher values indicate stronger non-potential magnetic configurations.
* Useful for characterizing magnetic complexity.

**Unit:** A

---

### 6. `ABSNJZH`

**Full meaning:** Absolute Net Current Helicity

Measures the absolute value of the net current helicity.

* Describes magnetic twist associated with electric currents.
* Higher values indicate stronger non-potential magnetic structure.
* The absolute value removes the distinction between positive and negative helicity.

**Unit:** G²/m

---

### 7. `SAVNCPP`

**Full meaning:** Sum of Absolute Value of Net Currents Per Polarity

Measures the absolute net electric currents associated with the positive and negative magnetic polarities.

* Characterizes electric-current imbalance.
* Higher values indicate stronger current structures.
* Useful for measuring magnetic non-potentiality.

**Unit:** A

---

### 8. `USFLUX`

**Full meaning:** Total Unsigned Magnetic Flux

Measures the total unsigned magnetic flux of the active region.

* Represents the total magnetic flux regardless of polarity.
* Higher values generally indicate larger or more strongly magnetized active regions.
* One of the fundamental measures of active-region magnetic strength.

**Unit:** Mx (Maxwell)

---

### 9. `TOTFZ`

**Full meaning:** Total Vertical Lorentz Force

Measures the total vertical component of the Lorentz force.

* Describes magnetic forces acting in the vertical direction.
* Provides information about departures from magnetic equilibrium.
* Related to the non-potential nature of the magnetic field.

**Unit:** dyne

---

### 10. `MEANPOT`

**Full meaning:** Mean Magnetic Free Energy Density

Represents the mean magnetic free-energy-related quantity across the active region.

* Describes the average magnetic energy available beyond the potential-field configuration.
* Higher values indicate more energetic and non-potential magnetic configurations.

**Unit:** erg/cm³

---

### 11. `EPSX`

**Full meaning:** X-component of Magnetic Field Energy/Stress

Represents the X-direction component associated with the magnetic-field energy or stress.

* Describes the magnetic contribution in the X direction.
* Used with `EPSY` and `EPSZ` to characterize the magnetic stress/energy distribution.

---

### 12. `EPSY`

**Full meaning:** Y-component of Magnetic Field Energy/Stress

Represents the Y-direction component associated with the magnetic-field energy or stress.

* Describes the magnetic contribution in the Y direction.
* Used with `EPSX` and `EPSZ` to characterize the magnetic stress/energy distribution.

---

### 13. `EPSZ`

**Full meaning:** Z-component of Magnetic Field Energy/Stress

Represents the Z-direction component associated with the magnetic-field energy or stress.

* Describes the magnetic contribution in the vertical direction.
* Used with `EPSX` and `EPSY` to characterize the magnetic stress/energy distribution.

---

### 14. `MEANSHR`

**Full meaning:** Mean Shear Angle

Measures the average shear angle between the observed magnetic field and the corresponding potential magnetic field.

* Indicates how far the magnetic field deviates from a potential-field configuration.
* Larger values indicate stronger magnetic shear.
* Strong magnetic shear is associated with increased magnetic complexity.

**Unit:** Degrees (°)

---

### 15. `SHRGT45`

**Full meaning:** Fraction of Area with Shear Angle Greater Than 45°

Measures the fraction of the active-region area where the magnetic shear angle exceeds 45°.

* Indicates how much of the active region contains strongly sheared magnetic fields.
* Higher values indicate a larger area of highly non-potential magnetic structure.

**Unit:** Dimensionless fraction

---

### 16. `MEANGAM`

**Full meaning:** Mean Magnetic Field Inclination Angle

Measures the average inclination angle of the magnetic field relative to the solar surface.

* Describes the average orientation of the magnetic field.
* Provides information about whether the magnetic field is more horizontal or vertical.

**Unit:** Degrees (°)

---

### 17. `MEANGBT`

**Full meaning:** Mean Gradient of the Total Magnetic Field

Measures the average spatial gradient of the total magnetic-field strength.

* Describes how rapidly the magnetic-field magnitude changes across the active region.
* Larger values indicate stronger spatial variations in the magnetic field.
* Useful for characterizing magnetic complexity.

**Unit:** G/cm

---

### 18. `MEANGBZ`

**Full meaning:** Mean Gradient of the Vertical Magnetic Field

Measures the average spatial gradient of the vertical magnetic-field component.

* Describes how rapidly the vertical magnetic field changes spatially.
* Strong gradients are often associated with complex magnetic structures and polarity inversion lines.

**Unit:** G/cm

---

### 19. `MEANGBH`

**Full meaning:** Mean Gradient of the Horizontal Magnetic Field

Measures the average spatial gradient of the horizontal magnetic-field component.

* Describes how rapidly the horizontal magnetic field changes across the active region.
* Provides information about the spatial complexity of the horizontal field.

**Unit:** G/cm

---

### 20. `MEANJZH`

**Full meaning:** Mean Vertical Current Helicity

Measures the mean current helicity associated with the vertical electric current.

* Describes magnetic twist associated with electric currents.
* The sign indicates the handedness of the magnetic structure.
* Useful for characterizing magnetic non-potentiality.

**Unit:** G²/m

---

### 21. `TOTFY`

**Full meaning:** Total Y-component of the Lorentz Force

Measures the total Lorentz-force component in the Y direction.

* Describes the net magnetic force acting in the Y direction.
* Part of the three-dimensional Lorentz-force characterization.

**Unit:** dyne

---

### 22. `MEANJZD`

**Full meaning:** Mean Vertical Current Density

Measures the average vertical electric current density in the active region.

* Provides information about electric currents flowing through the photosphere.
* Higher values indicate stronger current structures.
* Useful for identifying non-potential magnetic configurations.

**Unit:** mA/m²

---

### 23. `MEANALP`

**Full meaning:** Mean Alpha Parameter

Measures the mean force-free parameter (`α`) associated with the magnetic field.

The force-free relationship can be represented approximately as:

[
\nabla \times B = \alpha B
]

* Indicates the degree of magnetic twist.
* The sign of `α` indicates the handedness of the magnetic structure.
* Larger absolute values indicate stronger magnetic non-potentiality.

**Unit:** 1/m

---

### 24. `TOTFX`

**Full meaning:** Total X-component of the Lorentz Force

Measures the total Lorentz-force component in the X direction.

* Describes the net magnetic force acting in the X direction.
* Used together with `TOTFY` and `TOTFZ` to characterize three-dimensional magnetic forces.

**Unit:** dyne

---

# Summary Table

| Attribute | Meaning                                | General Interpretation               |
| --------- | -------------------------------------- | ------------------------------------ |
| `R_VALUE` | Schrijver's R Parameter                | Strong magnetic-gradient/MPIL flux   |
| `TOTUSJH` | Total Unsigned Current Helicity        | Magnetic twist/current complexity    |
| `TOTBSQ`  | Total Magnetic Field Squared           | Overall magnetic-field strength      |
| `TOTPOT`  | Total Magnetic Free Energy Density     | Stored non-potential magnetic energy |
| `TOTUSJZ` | Total Unsigned Vertical Current        | Total electric-current strength      |
| `ABSNJZH` | Absolute Net Current Helicity          | Magnetic twist/non-potentiality      |
| `SAVNCPP` | Sum Absolute Net Currents Per Polarity | Current imbalance                    |
| `USFLUX`  | Total Unsigned Magnetic Flux           | Total magnetic flux                  |
| `TOTFZ`   | Total Vertical Lorentz Force           | Vertical magnetic force              |
| `MEANPOT` | Mean Magnetic Free Energy Density      | Average available magnetic energy    |
| `EPSX`    | X-component of magnetic stress/energy  | X-direction magnetic contribution    |
| `EPSY`    | Y-component of magnetic stress/energy  | Y-direction magnetic contribution    |
| `EPSZ`    | Z-component of magnetic stress/energy  | Z-direction magnetic contribution    |
| `MEANSHR` | Mean Shear Angle                       | Average magnetic-field shear         |
| `SHRGT45` | Area with Shear > 45°                  | Strongly sheared-field area          |
| `MEANGAM` | Mean Magnetic Inclination Angle        | Average field orientation            |
| `MEANGBT` | Mean Gradient of Total Field           | Total-field spatial gradient         |
| `MEANGBZ` | Mean Gradient of Vertical Field        | Vertical-field spatial gradient      |
| `MEANGBH` | Mean Gradient of Horizontal Field      | Horizontal-field spatial gradient    |
| `MEANJZH` | Mean Vertical Current Helicity         | Average current helicity             |
| `TOTFY`   | Total Y Lorentz Force                  | Y-direction magnetic force           |
| `MEANJZD` | Mean Vertical Current Density          | Average electric-current density     |
| `MEANALP` | Mean Alpha Parameter                   | Magnetic twist/current parameter     |
| `TOTFX`   | Total X Lorentz Force                  | X-direction magnetic force           |

# Dataset Structure

Each sample is a **Multivariate Time Series (MVTS)** containing 24 attributes across multiple timestamps.

```text
Sample
│
├── Timestamp 1 → 24 attributes
├── Timestamp 2 → 24 attributes
├── Timestamp 3 → 24 attributes
├── ...
└── Timestamp N → 24 attributes
                     │
                     └── Label: 0 or 1
```

The input data therefore has the general shape:

```text
(num_samples, num_timestamps, 24)
```

while the corresponding labels have the shape:

```text
(num_samples,)
```

The 24 attributes describe different aspects of the solar magnetic field, including **magnetic flux, magnetic-field strength, magnetic gradients, electric currents, magnetic helicity, magnetic shear, magnetic energy, and Lorentz forces**.

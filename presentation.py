import numpy as np
from utils import setup_logger

import config

logger = setup_logger("PresentationEngine")

class SolarPresentationEngine:
    """
    Translates numeric probability arrays into executive-level space weather threat reports.
    """
    def __init__(self):
        # NOAA Flare Definitions and Threat Metadata
        if config.NUM_CLASSES == 2:
            self.class_metadata = {
                0: {
                    "class_name": "No Flare / Background",
                    "hazard_level": "LOW / QUIET",
                    "color_code": "GREEN",
                    "summary": "Solar active region exhibits normal background conditions.",
                    "operational_impact": "None. Standard satellite and radio operations remain unaffected."
                },
                1: {
                    "class_name": "M/X-Class Solar Flare",
                    "hazard_level": "ELEVATED HAZARD",
                    "color_code": "ORANGE/RED",
                    "summary": "Significant solar flare event predicted (M or X class).",
                    "operational_impact": "Potential HF radio blackouts; satellite operations and GPS accuracy may be affected."
                }
            }
        elif config.NUM_CLASSES == 3:
            self.class_metadata = {
                0: {
                    "class_name": "No Flare / Background",
                    "hazard_level": "LOW / QUIET",
                    "color_code": "GREEN",
                    "summary": "Solar active region exhibits normal background conditions.",
                    "operational_impact": "None. Standard satellite and radio operations remain unaffected."
                },
                1: {
                    "class_name": "M-Class Solar Flare",
                    "hazard_level": "MODERATE HAZARD",
                    "color_code": "ORANGE",
                    "summary": "Significant flare event likely. Moderate magnetic energy releases expected.",
                    "operational_impact": "Brief high-frequency radio blackouts on the sunlit side of Earth; potential minor navigation degraded precision."
                },
                2: {
                    "class_name": "X-Class Solar Flare",
                    "hazard_level": "CRITICAL HAZARD",
                    "color_code": "RED",
                    "summary": "Major solar flare event predicted! Extreme electromagnetic radiation emission.",
                    "operational_impact": "Wide-area HF radio blackouts for hours; high risk of satellite surface charging and aviation communications disruption."
                }
            }
        else:
            self.class_metadata = {
                0: {
                    "class_name": "No Flare / Background",
                    "hazard_level": "LOW / QUIET",
                    "color_code": "GREEN",
                    "summary": "Solar active region exhibits normal background conditions.",
                    "operational_impact": "None. Standard satellite and radio operations remain unaffected."
                },
                1: {
                    "class_name": "C-Class Solar Flare",
                    "hazard_level": "MINOR",
                    "color_code": "YELLOW",
                    "summary": "Minor solar flare activity predicted within the time window.",
                    "operational_impact": "Weak or negligible impacts on high-frequency (HF) radio signals at high latitudes."
                },
                2: {
                    "class_name": "M-Class Solar Flare",
                    "hazard_level": "MODERATE HAZARD",
                    "color_code": "ORANGE",
                    "summary": "Significant flare event likely. Moderate magnetic energy releases expected.",
                    "operational_impact": "Brief high-frequency radio blackouts on the sunlit side of Earth; potential minor navigation degraded precision."
                },
                3: {
                    "class_name": "X-Class Solar Flare",
                    "hazard_level": "CRITICAL HAZARD",
                    "color_code": "RED",
                    "summary": "Major solar flare event predicted! Extreme electromagnetic radiation emission.",
                    "operational_impact": "Wide-area HF radio blackouts for hours; high risk of satellite surface charging and aviation communications disruption."
                }
            }

    def format_single_report(self, sample_idx, probabilities, model_name="Model"):
        """
        Formats a single observation's prediction array into a clean narrative report.
        """
        pred_class = int(np.argmax(probabilities))
        confidence = probabilities[pred_class] * 100.0
        meta = self.class_metadata[pred_class]

        report = f"""
================================================================================
             SOLAR FLARE SPACE WEATHER FORECAST REPORT ({model_name.upper()})
================================================================================
Sample Index            : #{sample_idx}
Predicted Hazard Level  : {meta['hazard_level']} [{meta['color_code']}]
Primary Flare Category  : {meta['class_name']}
Model Confidence        : {confidence:.2f}%

EXECUTIVE SUMMARY:
  {meta['summary']}

OPERATIONAL IMPACT ASSESSMENT:
  {meta['operational_impact']}

PROBABILITY DISTRIBUTION BREAKDOWN:
  - Class 0 (Quiet / Background) : {probabilities[0]*100:6.2f}%
"""
        if len(probabilities) == 2:
            report += f"""  - Class 1 (M/X-Class Flare)  : {probabilities[1]*100:6.2f}%
================================================================================
"""
        elif len(probabilities) == 3:
            report += f"""  - Class 1 (M-Class Flare)     : {probabilities[1]*100:6.2f}%
  - Class 2 (X-Class Flare)     : {probabilities[2]*100:6.2f}%
================================================================================
"""
        else:
            report += f"""  - Class 1 (C-Class Flare)     : {probabilities[1]*100:6.2f}%
  - Class 2 (M-Class Flare)     : {probabilities[2]*100:6.2f}%
  - Class 3 (X-Class Flare)     : {probabilities[3]*100:6.2f}%
================================================================================
"""
        return report

    def compare_models_report(self, sample_idx, ml_probs, dl_probs, ensemble_probs=None):
        """
        Generates a comparative report contrasting models side by side.
        """
        ml_pred = int(np.argmax(ml_probs))
        dl_pred = int(np.argmax(dl_probs))
        ens_pred = int(np.argmax(ensemble_probs)) if ensemble_probs is not None else None
        
        ml_conf = ml_probs[ml_pred] * 100.0
        dl_conf = dl_probs[dl_pred] * 100.0
        ens_conf = ensemble_probs[ens_pred] * 100.0 if ensemble_probs is not None else 0.0

        ml_class_name = self.class_metadata[ml_pred]["class_name"]
        dl_class_name = self.class_metadata[dl_pred]["class_name"]
        ens_class_name = self.class_metadata[ens_pred]["class_name"] if ens_pred is not None else "N/A"

        ens_col_header = " | Soft-Voting Ensemble" if ensemble_probs is not None else ""
        ens_pred_str = f" | {ens_class_name:<20}" if ensemble_probs is not None else ""
        ens_conf_str = f" | {ens_conf:6.2f}%              " if ensemble_probs is not None else ""

        comparison_report = f"""
================================================================================
           SIDE-BY-SIDE FORECAST COMPARISON (SAMPLE #{sample_idx})
================================================================================
Metric / Model           | XGBoost (Machine Learning) | PyTorch (CNN-LSTM DL){ens_col_header}
--------------------------------------------------------------------------------
Predicted Event          | {ml_class_name:<26} | {dl_class_name:<24}{ens_pred_str}
Confidence Score         | {ml_conf:6.2f}%                    | {dl_conf:6.2f}%              {ens_conf_str}
"""
        if len(ml_probs) == 2:
            ens_f = f" | {ensemble_probs[1]*100:6.2f}%              " if ensemble_probs is not None else ""
            ens_q = f" | {ensemble_probs[0]*100:6.2f}%              " if ensemble_probs is not None else ""
            comparison_report += f"""Flare Prob   (Class 1)   | {ml_probs[1]*100:6.2f}%                    | {dl_probs[1]*100:6.2f}%              {ens_f}
Quiet Prob   (Class 0)   | {ml_probs[0]*100:6.2f}%                    | {dl_probs[0]*100:6.2f}%              {ens_q}
================================================================================
"""
        elif len(ml_probs) == 3:
            ens_x = f" | {ensemble_probs[2]*100:6.2f}%              " if ensemble_probs is not None else ""
            ens_m = f" | {ensemble_probs[1]*100:6.2f}%              " if ensemble_probs is not None else ""
            ens_q = f" | {ensemble_probs[0]*100:6.2f}%              " if ensemble_probs is not None else ""
            comparison_report += f"""X-Class Prob (Class 2)   | {ml_probs[2]*100:6.2f}%                    | {dl_probs[2]*100:6.2f}%              {ens_x}
M-Class Prob (Class 1)   | {ml_probs[1]*100:6.2f}%                    | {dl_probs[1]*100:6.2f}%              {ens_m}
Quiet Prob   (Class 0)   | {ml_probs[0]*100:6.2f}%                    | {dl_probs[0]*100:6.2f}%              {ens_q}
================================================================================
"""
        else:
            ens_x = f" | {ensemble_probs[3]*100:6.2f}%              " if ensemble_probs is not None else ""
            ens_m = f" | {ensemble_probs[2]*100:6.2f}%              " if ensemble_probs is not None else ""
            ens_c = f" | {ensemble_probs[1]*100:6.2f}%              " if ensemble_probs is not None else ""
            ens_q = f" | {ensemble_probs[0]*100:6.2f}%              " if ensemble_probs is not None else ""
            comparison_report += f"""X-Class Prob (Class 3)   | {ml_probs[3]*100:6.2f}%                    | {dl_probs[3]*100:6.2f}%              {ens_x}
M-Class Prob (Class 2)   | {ml_probs[2]*100:6.2f}%                    | {dl_probs[2]*100:6.2f}%              {ens_m}
C-Class Prob (Class 1)   | {ml_probs[1]*100:6.2f}%                    | {dl_probs[1]*100:6.2f}%              {ens_c}
Quiet Prob   (Class 0)   | {ml_probs[0]*100:6.2f}%                    | {dl_probs[0]*100:6.2f}%              {ens_q}
================================================================================
"""
        return comparison_report

if __name__ == "__main__":
    # Test presentation format with sample output
    engine = SolarPresentationEngine()
    dummy_ml_p = np.array([0.05, 0.15, 0.70, 0.10])
    dummy_dl_p = np.array([0.02, 0.08, 0.85, 0.05])
    dummy_ens_p = np.array([0.03, 0.10, 0.80, 0.07])
    
    print(engine.format_single_report(10, dummy_dl_p, "PyTorch CNN-LSTM"))
    print(engine.compare_models_report(10, dummy_ml_p, dummy_dl_p, dummy_ens_p))
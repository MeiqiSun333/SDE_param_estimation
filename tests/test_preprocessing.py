"""Tests for emotion data preprocessing."""

import numpy as np
import pandas as pd

from src.data.preprocessing import EmotionDataProcessor


def _row(**overrides):
    row = {
        "UUID": "subject-1",
        "SAD_ES": 0,
        "STR_ES": 1,
        "SITMOD_ES": 2,
        "DIST_ES": 3,
        "REAP1_ES": 4,
        "RUM_ES": 5,
        "EMOCOPE_ES": 6,
        "time_hours": 0.0,
    }
    row.update(overrides)
    return row


def test_zero_is_preserved_as_a_valid_value():
    processor = EmotionDataProcessor()
    result = processor.prepare_modeling_data(pd.DataFrame([_row()]))

    assert len(result) == 1
    assert result.iloc[0]["SAD"] == 0


def test_rows_with_any_missing_emotion_are_removed():
    processor = EmotionDataProcessor()
    data = pd.DataFrame([_row(), _row(UUID="subject-2", RUM_ES=np.nan)])

    result = processor.prepare_modeling_data(data)

    assert result["UUID"].tolist() == ["subject-1"]


def test_pipeline_creates_output_directory(tmp_path):
    processor = EmotionDataProcessor()
    raw_path = tmp_path / "raw.csv"
    output = tmp_path / "nested" / "processed.csv"
    raw_row = _row()
    raw_row.pop("time_hours")
    raw_row.update({
        "Date_Local": "01/01/2026",
        "Time_Local": "12:00:00",
        "dataset": "test",
    })
    pd.DataFrame([raw_row]).to_csv(raw_path, index=False)

    processor.process_pipeline(str(raw_path), str(output))

    assert output.exists()


def test_delta_t_is_recomputed_after_missing_rows_are_removed():
    processor = EmotionDataProcessor()
    data = pd.DataFrame(
        [
            _row(time_hours=0.0),
            _row(time_hours=1.0, RUM_ES=np.nan),
            _row(time_hours=3.0),
        ]
    )

    result = processor.prepare_modeling_data(data)

    assert np.isnan(result.iloc[0]["delta_t"])
    assert result.iloc[1]["delta_t"] == 3.0

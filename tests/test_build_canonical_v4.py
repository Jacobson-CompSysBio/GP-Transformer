import unittest

import numpy as np
import pandas as pd

from scripts.build_canonical_v4 import (
    TARGET,
    WEATHER_COLUMNS,
    _aggregate_weather,
    _row_table,
)


class CanonicalV4WeatherTests(unittest.TestCase):
    def test_named_accumulations_never_borrow_another_weather_column(self):
        dates = pd.to_datetime(["2023-04-01", "2023-04-02", "2023-04-03"])
        weather = pd.DataFrame(
            {
                "Env": ["AAH1_2023"] * 3,
                "Date": [20230401, 20230402, 20230403],
            }
        )
        for index, column in enumerate(WEATHER_COLUMNS):
            weather[column] = np.array([index + 1.0, index + 2.0, index + 3.0])
        weather["ALLSKY_SFC_SW_DWN"] = [100.0, 200.0, 300.0]
        weather.loc[1, "QV2M"] = np.nan
        windows = pd.DataFrame(
            {
                "window_start_current": [dates[0]],
                "window_end": [dates[-1]],
            },
            index=["AAH1_2023"],
        )

        result, provenance = _aggregate_weather(
            weather, windows, "window_start_current", "unit-test-window"
        )

        self.assertEqual(result.loc["AAH1_2023", "QV2M_acum"], 12.0)
        self.assertEqual(
            result.loc["AAH1_2023", "ALLSKY_SFC_SW_DWN_acum"], 600.0
        )
        self.assertEqual(result.loc["AAH1_2023", "meta__weather_missing_days__QV2M"], 1)
        self.assertEqual(result.loc["AAH1_2023", "num_days"], 3.0)
        self.assertEqual(provenance["aggregation"], "named-column min/max/mean/sum(min_count=1)")


class CanonicalV4RowTests(unittest.TestCase):
    def test_physical_replicates_keep_distinct_row_indices_and_source_indices(self):
        raw = pd.DataFrame(
            {
                "source_row_index": [8, 11],
                "Env": ["TXH1-extra_2021", "TXH1-extra_2021"],
                "Hybrid": ["A/B", "A/B"],
                TARGET: [1.0, 3.0],
            }
        )
        rows = _row_table(raw, "complete", "unit rows")
        self.assertEqual(rows["row_index"].tolist(), [0, 1])
        self.assertEqual(rows["source_row_index"].tolist(), [8, 11])
        self.assertEqual(rows["id"].tolist(), ["TXH1-extra_2021-A/B"] * 2)
        self.assertEqual(rows["cell_replicate_count"].tolist(), [2, 2])
        self.assertEqual(rows["cell_yield_variance"].tolist(), [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()

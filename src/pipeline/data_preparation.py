"""Data preparation module for splitting and persisting dataset."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipeline.config_loader import load_params

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreparer:
    """Class responsible for splitting dataset into train and test sets."""

    def __init__(
        self, seed: int = 42, test_size: float = 0.2, target_col: str = "Revenue"
    ) -> None:
        """Initialize data preparer configuration."""
        self.seed = seed
        self.test_size = test_size
        self.target_col = target_col

    def load_data(self, file_path: str | Path) -> pd.DataFrame:
        """Load dataset from csv file.

        Args:
            file_path: Path to raw dataset.

        Returns:
            DataFrame with raw data.
        """
        path = Path(file_path)
        logger.info("Loading raw data from %s", path)
        return pd.read_csv(path)

    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Perform stratified train/test split.

        Args:
            df: Full raw dataframe.

        Returns:
            Tuple with train and test DataFrames.
        """
        logger.info(
            "Performing stratified train/test split (test_size=%.2f)", self.test_size
        )
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=df[self.target_col],
        )
        return train_df, test_df

    def save_data(self, df: pd.DataFrame, output_path: str | Path) -> None:
        """Save dataframe to CSV format.

        Args:
            df: DataFrame to save.
            output_path: Target path for file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved dataset to %s (%d rows)", path, len(df))

    def run(self, raw_path: str | Path, output_dir: str | Path) -> None:
        """Execute full data preparation pipeline.

        Args:
            raw_path: Raw dataset path.
            output_dir: Directory where train and test sets will be saved.
        """
        df = self.load_data(raw_path)
        train_df, test_df = self.split_data(df)
        out_dir = Path(output_dir)
        self.save_data(train_df, out_dir / "train.csv")
        self.save_data(test_df, out_dir / "test.csv")


def main() -> None:
    """Entry point for data preparation stage."""
    params = load_params()
    seed = int(params.get("seed", 42))
    prep_cfg = params.get("prepare", {})

    preparer = DataPreparer(
        seed=seed,
        test_size=float(prep_cfg.get("test_size", 0.2)),
        target_col=str(prep_cfg.get("target_col", "Revenue")),
    )
    preparer.run(
        raw_path=prep_cfg.get(
            "raw_data_path", "data/raw/online_shoppers_intention.csv"
        ),
        output_dir=prep_cfg.get("processed_dir", "data/processed"),
    )


if __name__ == "__main__":
    main()

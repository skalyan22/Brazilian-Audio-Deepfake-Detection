import os

import pandas as pd

TSV_PATH = "datasets/portuguese/common_voice/validated.tsv"
OUT_DIR = "experiments/accent_audit"

ACCENT_TO_REGION = {
    "Paulista": "Southeast-SP",
    "Interior Paulista": "Southeast-SP",
    "Brasil, São Paulo": "Southeast-SP",
    "Paulista,Interior Paulista,Caipira ,Ribeirão Preto": "Southeast-SP",
    'paulista com enfase no "R"': "Southeast-SP",
    "Interior of São Paulo": "Southeast-SP",
    "native Brazillian Portuguese - city of Sao Paulo": "Southeast-SP",
    "Paulista,Brasileiro": "Southeast-SP",
    "Brasil, interior de São Paulo": "Southeast-SP",
    "Paulista,Paulistano,São Paulo": "Southeast-SP",
    "Paulista do interior": "Southeast-SP",
    "Carioca": "Southeast-RJ",
    "Carioca,Rio de Janeiro,RJ": "Southeast-RJ",
    "Mineiro": "Southeast-MG",
    "Mineiro,Minas": "Southeast-MG",
    "Nordestino": "Northeast",
    "Nordestino,Padrão": "Northeast",
    "Fortaleza,Mistura": "Northeast",
    "Cearense": "Northeast",
    "baiano,whistled s": "Northeast",
    "Manauara,Noroeste": "North",
    "Centro-oeste": "Center-West",
    "meu sotaque é tradicional da região sul do brasil": "South",
    "Manezinho (Florianópolis - SC)": "South",
    "Nativo,catarinense": "South",
    "Brasileiro,Sulista,catarinense,tubaronense": "South",
    "southern brazil accent": "South",
    "Normal,Um pouco sulista": "South",
    "gaúcho - interior": "South",
    "Paulista, Carioca e Sulista": "mixed_br",
    "Mineiro,Carioca": "mixed_br",
    "Mineiro,Maranhense": "mixed_br",
    "Brasileiro": "br_unspecified",
    "Brazilian": "br_unspecified",
    'Quase sem sotaque, respeitando o som original do "R" arrastado. ,Dificuldades na dicção de encontros consonantais:  "dr", "tr".': "br_unspecified",
    "Jovem,Duriense": "european_pt",
    "Porto": "european_pt",
    "Porto,Minhoto Central": "european_pt",
    "native": "other_ambiguous",
    "Français": "other_ambiguous",
}

REGION_ORDER = [
    "Southeast-SP", "Southeast-RJ", "Southeast-MG", "Northeast",
    "South", "North", "Center-West", "mixed_br", "br_unspecified",
    "european_pt", "other_ambiguous", "(no accent label)",
]

BRAZILIAN_REGIONS = [
    "Southeast-SP", "Southeast-RJ", "Southeast-MG", "Northeast",
    "South", "North", "Center-West",
]


class AccentNormalizer:
    @staticmethod
    def to_region(accent):
        if pd.isna(accent):
            return "(no accent label)"
        return ACCENT_TO_REGION.get(accent, "other_ambiguous")

    @staticmethod
    def unmapped_values(df):
        return set(df["accents"].dropna().unique()) - set(ACCENT_TO_REGION)


class CoverageReport:
    def __init__(self, df):
        self.df = df
        self.n = len(df)

    def region_coverage(self):
        coverage = (
            self.df.groupby("region")
            .agg(clips=("path", "count"), speakers=("client_id", "nunique"))
            .reindex(REGION_ORDER)
            .dropna(how="all")
            .astype(int)
        )
        coverage["pct_clips"] = (100 * coverage["clips"] / self.n).round(1)
        return coverage

    def crosstab_by_region(self, column):
        return (
            pd.crosstab(self.df["region"], self.df[column].fillna("(missing)"))
            .reindex(REGION_ORDER)
            .dropna(how="all")
            .astype(int)
        )

    def metadata_availability(self):
        return {
            "accent/region label": self.df["accents"].notna().sum(),
            "gender label": self.df["gender"].notna().sum(),
            "age label": self.df["age"].notna().sum(),
            "variant label": self.df["variant"].notna().sum(),
        }

    def to_markdown(self):
        coverage = self.region_coverage()
        gender = self.crosstab_by_region("gender")
        age = self.crosstab_by_region("age")
        availability = self.metadata_availability()
        br_labeled = self.df[self.df["region"].isin(BRAZILIAN_REGIONS)]

        lines = ["# BrazilianDF - Accent Coverage Audit (Experiment 4)\n"]
        lines.append(
            f"**Source**: Common Voice PT `validated.tsv` ({self.n:,} clips, "
            f"{self.df['client_id'].nunique():,} unique speakers)\n")
        lines.append("## Metadata availability\n")
        lines.append("| Field | Clips with label | % of corpus |")
        lines.append("|---|---|---|")
        for field, count in availability.items():
            lines.append(f"| {field} | {count:,} | {100 * count / self.n:.1f}% |")
        lines.append("")
        lines.append("## Accent / region coverage table\n")
        lines.append("| Region | Clips | Unique speakers | % of clips |")
        lines.append("|---|---|---|---|")
        for region, row in coverage.iterrows():
            lines.append(f"| {region} | {int(row['clips']):,} | "
                         f"{int(row['speakers']):,} | {row['pct_clips']}% |")
        lines.append("")
        for title, table in (("Gender", gender), ("Age", age)):
            lines.append(f"## {title} by region (clips)\n")
            lines.append("| Region | " + " | ".join(table.columns) + " |")
            lines.append("|---" * (len(table.columns) + 1) + "|")
            for region, row in table.iterrows():
                lines.append(f"| {region} | " +
                             " | ".join(str(v) for v in row.values) + " |")
            lines.append("")
        lines.append("## Key findings\n")
        n_labeled = availability["accent/region label"]
        lines.append(
            f"- Only **{n_labeled:,} / {self.n:,} clips "
            f"({100 * n_labeled / self.n:.1f}%)** carry any accent label; "
            f"of these, **{len(br_labeled):,} clips** map to a specific Brazilian "
            f"macro-region across **{br_labeled['client_id'].nunique()} speakers**.")
        lines.append(
            "- Accent strings are free text and were manually normalized to "
            "IBGE-style macro-regions; see `ACCENT_TO_REGION`.")
        n_european = int((self.df["variant"] == "Portuguese (Portugal)").sum())
        lines.append(
            f"- The `variant` column identifies {n_european} European Portuguese "
            "clips which should be excluded (or held out) for a PT-BR benchmark.")
        lines.append(
            "- Speaker counts per region are the binding constraint for fair, "
            "speaker-disjoint train/test splits - see table above.")
        lines.append("")
        return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(TSV_PATH, sep="\t")

    unmapped = AccentNormalizer.unmapped_values(df)
    if unmapped:
        print("WARNING - unmapped accent strings (assigned other_ambiguous):")
        for value in unmapped:
            print("  ", value)

    df["region"] = df["accents"].map(AccentNormalizer.to_region)

    report = CoverageReport(df)
    report.region_coverage().to_csv(os.path.join(OUT_DIR, "accent_coverage.csv"))
    df[["path", "client_id", "accents", "region", "gender", "age", "variant"]].to_csv(
        os.path.join(OUT_DIR, "clips_with_region.csv"), index=False)

    markdown = report.to_markdown()
    report_path = os.path.join(OUT_DIR, "ACCENT_COVERAGE.md")
    with open(report_path, "w") as f:
        f.write(markdown)

    print(markdown)
    print(f"\nWritten: {report_path}")


if __name__ == "__main__":
    main()

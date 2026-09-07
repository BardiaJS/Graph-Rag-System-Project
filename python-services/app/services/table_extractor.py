from typing import Any, Dict, List

import camelot


class TableExtractor:

    def extract_tables(
        self,
        pdf_path: str,
    ) -> List[Dict[str, Any]]:

        results: List[Dict[str, Any]] = []

        tables = None

        # ------------------------------------------------------
        # اول lattice
        # ------------------------------------------------------

        try:

            tables = camelot.read_pdf(
                pdf_path,
                pages="all",
                flavor="lattice",
            )

        except Exception as e:

            print(
                f"   ⚠️ Camelot lattice failed: {e}"
            )

            tables = None

        # ------------------------------------------------------
        # اگر lattice جواب نداد -> stream
        # ------------------------------------------------------

        if not tables or len(tables) == 0:

            try:

                tables = camelot.read_pdf(
                    pdf_path,
                    pages="all",
                    flavor="stream",
                )

            except Exception as e:

                print(
                    f"   ⚠️ Camelot stream failed: {e}"
                )

                return []

        # ------------------------------------------------------
        # Convert tables
        # ------------------------------------------------------

        for index, table in enumerate(tables):

            try:

                df = table.df

                if df is None or df.empty:
                    continue

                headers = [
                    str(value).strip()
                    for value in df.iloc[0].tolist()
                ]

                normalized_headers = []

                for col_idx, header in enumerate(
                    headers
                ):

                    if not header:
                        header = (
                            f"column_{col_idx}"
                        )

                    normalized_headers.append(
                        header
                    )

                data = []

                for row_idx in range(
                    1,
                    len(df),
                ):

                    row = df.iloc[row_idx]

                    row_data = {}

                    for col_idx, header in enumerate(
                        normalized_headers
                    ):

                        if col_idx >= len(row):
                            continue

                        row_data[header] = str(
                            row.iloc[col_idx]
                        ).strip()

                    data.append(row_data)

                results.append(
                    {
                        "index": index,
                        "page": table.page,
                        "headers": normalized_headers,
                        "rows": len(data),
                        "data": data,
                        "html": df.to_html(
                            index=False
                        ),
                    }
                )

            except Exception as e:

                print(
                    f"   ⚠️ Failed processing "
                    f"table {index}: {e}"
                )

        print(
            f"📊 Extracted {len(results)} tables"
        )

        return results
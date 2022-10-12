import numpy as np
import os, io
import pandas as pd

def get_stats(dimension):
	if dimension == 5:
		benchmark_name = f"QP-{dimension}d"
	else:
		benchmark_name = f"QP-{dimension}d-5s"
	print(benchmark_name)

	tab_path = f"qp_data/{benchmark_name}.xlsx"
	sheet_names = ['success probability', 'physical runtime', 'time-to-solution (tts)']
	tab_data = {}
	for sheet_name in sheet_names:
		tab_data[sheet_name] = pd.read_excel(tab_path, sheet_name=sheet_name, index_col=0)
		tab_data[sheet_name].replace(np.inf, np.nan, inplace=True)

	stats = pd.DataFrame({'Stats':sheet_names})
	for c in tab_data[sheet_names[0]].columns:
		means = []
		stds = []
		for sheet_name in sheet_names:
			means.append(tab_data[sheet_name][c].mean(skipna=True))
			stds.append(tab_data[sheet_name][c].std(skipna=True))
		stats[c] = [f"{means[j]:.2e}({stds[j]:.2e})" for j in range(3)]
	
	return stats.set_index('Stats')

def save_stats_all(dimensions):
	
	save_excel_path = "qp_data"
	excel_name = "QP-stats-all.xlsx"
	with pd.ExcelWriter(os.path.join(save_excel_path, excel_name)) as writer:
		for dimension in dimensions:
			stats = get_stats(dimension)
			stats.to_excel(writer, sheet_name=f"{dimension}d")

	return None

if __name__ == "__main__":
	dimensions = [5, 50, 60, 75]
	save_stats_all(dimensions)
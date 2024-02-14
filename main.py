import pandas as pd
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import os
import seaborn as sns
import json

def find_critical_path_and_create_cct(df_group):
    trace_id, trace_df = df_group
    trace_df.sort_values(by='timestamp', inplace=True)
    trace_df['endtime'] = trace_df['timestamp'] + trace_df['rt']
    max_span_index = trace_df['endtime'].idxmax()
    max_span = trace_df.loc[max_span_index]

    cct = defaultdict(list)
    for index, row in trace_df.iterrows():
        cct[row['um']].append(row['dm'])

    critical_path = [max_span['um']]
    if max_span['dm'] not in critical_path:
        critical_path.append(max_span['dm'])
    response_times = [max_span['rt']]
    current_dm = max_span['dm']

    while True:
        next_spans = trace_df[(trace_df['um'] == current_dm) & (trace_df.index > max_span_index)]
        if next_spans.empty:
            break
        max_rt_span = next_spans.loc[next_spans['rt'].idxmax()]
        if max_rt_span['dm'] not in critical_path:
            critical_path.append(max_rt_span['dm'])
            response_times.append(max_rt_span['rt'])
            current_dm = max_rt_span['dm']
        max_span_index = max_rt_span.name

    critical_path_str = " --> ".join(critical_path)
    return trace_id, critical_path_str, response_times, cct

def process_file(file_path):
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Number of records: {df.shape[0]}")

    num_threads = min(10, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        grouped = df.groupby('traceid')
        results = list(executor.map(find_critical_path_and_create_cct, grouped))

    critical_paths = {trace_id: (critical_path_str, response_times) for trace_id, critical_path_str, response_times, _ in results}
    return critical_paths

def analyze_performance_variation(all_critical_paths):
    aggregated_data = defaultdict(lambda: defaultdict(lambda: {'times': [], 'count': 0}))

    for interval, paths in all_critical_paths.items():
        for trace_id, (path, rts) in paths.items():
            interactions = path.split(" --> ")
            for i in range(len(interactions) - 1):
                key = f"{interactions[i]} --> {interactions[i + 1]}"
                aggregated_data[key][interval]['times'].extend([rts[i]])
                aggregated_data[key][interval]['count'] += 1

    interaction_stats = {}
    for interaction, intervals in aggregated_data.items():
        for interval, data in intervals.items():
            if interaction not in interaction_stats:
                interaction_stats[interaction] = {'intervals': [], 'means': [], 'stds': [], 'counts': []}
            interaction_stats[interaction]['intervals'].append(interval)
            interaction_stats[interaction]['means'].append(np.mean(data['times']))
            interaction_stats[interaction]['stds'].append(np.std(data['times']))
            interaction_stats[interaction]['counts'].append(data['count'])

    # visualize_high_variation_interactions(interaction_stats, all_critical_paths)
    return interaction_stats

def visualize_high_variation_interactions(interaction_stats, all_critical_paths):
    output_directory = "output_plots"  # Adjusted to a more descriptive name
    os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist

    plot_count = 0  # Initialize a counter for the plot files
    for interaction, stats in interaction_stats.items():
        if any(std > 10 * np.mean(stats['stds']) for std in stats['stds']):
            fig, ax1 = plt.subplots(figsize=(10, 6))

            ax2 = ax1.twinx()
            bars = ax1.bar(range(len(stats['means'])), stats['means'], yerr=stats['stds'], capsize=5, color='g')
            ax2.plot(range(len(stats['counts'])), stats['counts'], 'b-')

            for bar, std in zip(bars, stats['stds']):
                height = bar.get_height()
                ax1.annotate(f'{std:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom')

            critical_path = "No path"
            for paths in all_critical_paths.values():
                for trace_id, (path, rts) in paths.items():
                    if interaction in path:
                        critical_path = path
                        break

            ax1.set_title(f"Interaction: {interaction} in critical path: {critical_path}")
            ax1.set_xticks(range(len(stats['means'])))
            ax1.set_xticklabels([f'{x * 3}-{(x + 1) * 3}min' for x in stats['intervals']], rotation=45)
            ax1.set_ylabel("Response Time (ms)", color='g')
            ax2.set_ylabel("Count", color='b')
            ax1.set_xlabel("Time Interval")

            plt.tight_layout()

            plot_file_name = os.path.join(output_directory, f"interaction_{plot_count}.png")
            plt.savefig(plot_file_name)
            print(f"Plot saved as {plot_file_name}")

            plt.close(fig)
            plot_count += 1

def summarize_insights(all_critical_paths, interaction_stats, output_file_path):
    all_critical_paths_file = output_file_path.replace('.txt', '_all_critical_paths.json')
    with open(all_critical_paths_file, 'w') as file:
        json.dump(all_critical_paths, file, indent=4)

    interaction_stats_file = output_file_path.replace('.txt', '_interaction_stats.json')
    with open(interaction_stats_file, 'w') as file:
        json.dump(interaction_stats, file, indent=4)

    unique_traces = sum(len(paths) for _, paths in all_critical_paths.items())
    unique_critical_paths = set(path for paths in all_critical_paths.values() for _, (path, _) in paths.items())
    high_variation_interactions = {interaction for interaction, stats in interaction_stats.items() if any(std > 10 * np.mean(stats['stds']) for std in stats['stds'])}

    with open(output_file_path, 'w') as file:
        file.write(f"Number of unique traces: {unique_traces}\n")
        file.write(f"Number of unique critical paths: {len(unique_critical_paths)}\n")
        file.write(f"High variation interactions: {len(high_variation_interactions)}\n\n")
        file.write("Detailed Interaction Stats:\n")
        for interaction, data in interaction_stats.items():
            file.write(f"Interaction: {interaction}, Mean RT: {np.mean(data['means']):.2f} ms, Std RT: {np.mean(data['stds']):.2f} ms, Count: {sum(data['counts'])}\n")

    print(f"Summary saved to {output_file_path}")

if __name__ == "__main__":
    all_critical_paths = defaultdict(dict)

    # This is a placeholder loop. Replace with actual file processing logic.
    for i in range(20):  # Adjust as needed
        file_path = f'file_{i}.csv'  # Adjust as needed
        critical_paths = process_file(file_path)
        all_critical_paths[i] = critical_paths

    interaction_stats = analyze_performance_variation(all_critical_paths)

    output_summary_file = "performance_analysis_summary.txt"
    summarize_insights(all_critical_paths, interaction_stats, output_summary_file)

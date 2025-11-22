"""
Diversity Factor Calculator for Power Distribution Systems

A substation supplies power by four feeders to its consumers.
This program calculates the diversity factor for different groups of consumers.

Diversity Factor = Sum of Individual Maximum Demands / Maximum Demand on the Group
"""


def calculate_diversity_factor(individual_demands, max_demand_on_group):
    """
    Calculate the diversity factor for a group of consumers.

    Args:
        individual_demands (list): List of individual maximum demands in kW
        max_demand_on_group (float): Maximum demand on the group in kW

    Returns:
        tuple: (sum_of_demands, diversity_factor)
    """
    sum_of_demands = sum(individual_demands)
    diversity_factor = sum_of_demands / max_demand_on_group
    return sum_of_demands, diversity_factor


def main():
    """
    Solve the diversity factor problem for the given substation.
    """
    print("=" * 70)
    print("DIVERSITY FACTOR CALCULATOR")
    print("=" * 70)
    print()

    # Feeder No. 1 data
    feeder1_consumers = [70, 90, 20, 50, 10, 20]  # kW
    feeder1_max_demand = 200  # kW

    print("FEEDER NO. 1")
    print("-" * 70)
    print(f"Individual consumer demands: {feeder1_consumers} kW")
    print(f"Number of consumers: {len(feeder1_consumers)}")

    sum_feeder1, df_feeder1 = calculate_diversity_factor(
        feeder1_consumers, feeder1_max_demand
    )

    print(f"Sum of individual maximum demands: {sum_feeder1} kW")
    print(f"Maximum demand on feeder: {feeder1_max_demand} kW")
    print(f"Diversity Factor = {sum_feeder1} / {feeder1_max_demand} = {df_feeder1:.4f}")
    print()

    # Feeder No. 2 data
    feeder2_consumers = [60, 40, 70, 30]  # kW
    feeder2_max_demand = 160  # kW

    print("FEEDER NO. 2")
    print("-" * 70)
    print(f"Individual consumer demands: {feeder2_consumers} kW")
    print(f"Number of consumers: {len(feeder2_consumers)}")

    sum_feeder2, df_feeder2 = calculate_diversity_factor(
        feeder2_consumers, feeder2_max_demand
    )

    print(f"Sum of individual maximum demands: {sum_feeder2} kW")
    print(f"Maximum demand on feeder: {feeder2_max_demand} kW")
    print(f"Diversity Factor = {sum_feeder2} / {feeder2_max_demand} = {df_feeder2:.4f}")
    print()

    # Four feeders data
    feeder_max_demands = [200, 160, 150, 200]  # kW for feeders 1, 2, 3, 4
    station_max_demand = 600  # kW

    print("FOUR FEEDERS")
    print("-" * 70)
    print(f"Maximum demands on feeders: {feeder_max_demands} kW")
    print(f"Number of feeders: {len(feeder_max_demands)}")

    sum_feeders, df_feeders = calculate_diversity_factor(
        feeder_max_demands, station_max_demand
    )

    print(f"Sum of feeder maximum demands: {sum_feeders} kW")
    print(f"Maximum demand on station: {station_max_demand} kW")
    print(f"Diversity Factor = {sum_feeders} / {station_max_demand} = {df_feeders:.4f}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    print(f"Diversity Factor for Feeder No. 1 consumers: {df_feeder1:.4f}")
    print(f"Diversity Factor for Feeder No. 2 consumers: {df_feeder2:.4f}")
    print(f"Diversity Factor for the Four Feeders:      {df_feeders:.4f}")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("A diversity factor > 1 indicates that the sum of individual maximum")
    print("demands is greater than the maximum demand on the group.")
    print("This occurs because all consumers do not reach their maximum demand")
    print("at the same time, allowing for more efficient use of the distribution")
    print("system.")
    print()
    print(f"Feeder 1: DF = {df_feeder1:.4f} - {'Good diversity' if df_feeder1 > 1 else 'Poor diversity'}")
    print(f"Feeder 2: DF = {df_feeder2:.4f} - {'Good diversity' if df_feeder2 > 1 else 'Poor diversity'}")
    print(f"Station:  DF = {df_feeders:.4f} - {'Good diversity' if df_feeders > 1 else 'Poor diversity'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

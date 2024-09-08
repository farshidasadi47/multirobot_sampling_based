# %%
########################################################################
# This files hold test function to test rrt method using subprocesses
# for parallel execution.
# This is part of sampling based motion planning for heterogeneous
# magnetic robots.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import argparse
import sys
import os
import json

try:
    from swarm import rrt
    import rrtparam
except ModuleNotFoundError:
    import rrtparam

    sys.path.append(os.path.abspath(".."))
    import rrt
except Exception as exc:
    print(type(exc).__name__, exc.args)
########## Main code ###################################################
planners = {
    "rrt3": rrtparam.rrt3,
    "rrt4": rrtparam.rrt4,
    "rrt5": rrtparam.rrt5,
    "rrt10big": rrtparam.rrt10big,
    "rrt10": rrtparam.rrt10,
}


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Evaluate RRT planning for given parameters."
        )
        parser.add_argument("--planner_name", type=str, required=True)
        parser.add_argument("--max_size", type=int, required=True)
        parser.add_argument("--params", type=str, required=True)
        #
        args = parser.parse_args()
        planner = planners.get(args.planner_name)
        max_size = args.max_size
        params = json.loads(args.params)
        if planner is None:
            print(
                f"Error: Planner '{args.planner_name}' not found.",
                file=sys.stderr,
            )
            sys.exit(1)
        # Call the planner anf get the results.
        result = planner(max_size, **params)
        # Print the result as JSON
        print(json.dumps(result))
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

import os
import argparse


def arguments():
	cli = argparse.ArgumentParser()

	# the environment to use (separate run for each arg)
	cli.add_argument(
		"--env",
		nargs="*",
		type=str,
		default=["sc2"],
	)

	# the config file to use (separate run for each arg)
	cli.add_argument(
		"--config",
		nargs="*",
		type=str,
		default=["qgnn"],
	)

	# extra config arguments (separate run for each arg)
	cli.add_argument(
		"--separate",
		nargs="*",
		type=str,
		default=[],
	)

	# extra config arguments (used for all runs)
	cli.add_argument(
		"--shared",
		nargs="*",
		type=str,
		default=[],
	)

	args = cli.parse_args()
	return args


def build_cmd(env, config, params):
	return "python src/main.py --config={config} --env-config={env} with {params}".format(env=env, config=config, params=" ".join(params))


def run_cmd(cmd):
	os.system(cmd)


if __name__ == '__main__':
	args = arguments()
	for env in args.env:
		for config in args.config:
			if len(args.separate) > 0:
				for param in args.separate:
					params = args.shared + [param]
					run_cmd(build_cmd(env, config, params))
			else:
				run_cmd(build_cmd(env, config, args.shared))



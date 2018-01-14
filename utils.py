import termcolor


def success(str):
  return termcolor.colored(str, 'green')


def warning(str):
  return termcolor.colored(str, 'yellow')


def log_args(args):
  print(warning('arguments:'))
  for key, value in vars(args).items():
    print(warning('\t{}:').format(key), value)

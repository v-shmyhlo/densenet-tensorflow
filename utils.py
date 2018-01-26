import termcolor


def success(str):
  return termcolor.colored(str, 'green')


def warning(str):
  return termcolor.colored(str, 'yellow')


def danger(str):
  return termcolor.colored(str, 'red')


def log_args(args):
  print(warning('arguments:'))
  for key, value in vars(args).items():
    print(warning('\t{}:').format(key), value)

import termcolor


def success(str):
  return termcolor.colored(str, 'green')


def warning(str):
  return termcolor.colored(str, 'yellow')

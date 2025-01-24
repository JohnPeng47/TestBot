import jinja2

a = """
{hello}

world!s
"""

print(jinja2.Template(a.format(hello="HELLO")).render(hello='Hello'))
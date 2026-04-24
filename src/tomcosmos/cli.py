import typer

from tomcosmos import __version__

app = typer.Typer(help="Solar system state simulator.", no_args_is_help=True)


@app.command()
def version() -> None:
    typer.echo(f"tomcosmos {__version__}")

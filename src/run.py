import melee
import sys
import os
import arg_parser
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path("../.env"))

ISO_PATH = os.getenv('ISO_PATH')
SLIPPI_PATH = os.getenv('SLIPPI_PATH')

console = melee.console.Console(
    path=SLIPPI_PATH,
    fullscreen=False
)

controller1 = melee.controller.Controller(
    console,
    port=1,
    type=melee.ControllerType.STANDARD
)

controller2 = melee.controller.Controller(
    console,
    port=2,
    type=melee.ControllerType.STANDARD
)

console.run(iso_path=ISO_PATH)
print("Connecting to console...")
if not console.connect():
    print("ERROR: Failed to connect to the console.")
    sys.exit(-1)
print("Console connected")

print("Connecting controllers to console...")
if not controller1.connect() or not controller2.connect():
    print("ERROR: Failed to connect the controllers.")
    sys.exit(-1)
print("Controllers connected")


while True:
    gamestate = console.step()
    if gamestate is None:
        continue


    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        pass
    else:
        melee.MenuHelper.menu_helper_simple(
            gamestate=gamestate,
            controller=controller1,
            character_selected=melee.Character.FOX,
            stage_selected=melee.Stage.YOSHIS_STORY,
            connect_code="",
            cpu_level=0,
            costume=0,
            autostart=True,
            swag=False
        )

        melee.MenuHelper.menu_helper_simple(
            gamestate=gamestate,
            controller=controller2,
            character_selected=melee.Character.BOWSER,
            stage_selected=melee.Stage.YOSHIS_STORY,
            connect_code="",
            cpu_level=6,
            costume=0,
            autostart=True,
            swag=False
        )
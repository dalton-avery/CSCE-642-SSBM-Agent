import melee
import sys
import Args

DOLPHIN_EXE_PATH, MELEE_ISO_PATH = Args.get_paths()

console = melee.console.Console(
    path=DOLPHIN_EXE_PATH,
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

console.run(iso_path=MELEE_ISO_PATH)
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
import json
import os
from pathlib import Path

from app.biometrics.face_verification import FaceVerifier

ROOT_DIR = Path(__file__).resolve().parents[1]
USERS_DIR = ROOT_DIR / "data" / "users"
USERS_DIR.mkdir(parents=True, exist_ok=True)

verifier = FaceVerifier(users_dir=str(USERS_DIR), model_name="ArcFace")


def _user_json_path(username: str) -> Path:
    return USERS_DIR / f"{username}.json"


def register_user(username: str, photo) -> bool:
    try:
        verifier.register_user(username, photo)
        return True
    except Exception as exc:
        print(f"Error registrando usuario: {exc}")
        return False


def login_user(username: str, photo) -> bool:
    if not os.path.exists(verifier._emb_path(username)):
        print("Usuario no registrado.")
        return False

    try:
        result = verifier.verify_user(photo, username)
        return result.get("match", False)
    except Exception as exc:
        print(f"Error verificando usuario: {exc}")
        return False


def user_exists(username: str) -> bool:
    """Devuelve True si el usuario esta registrado y tiene embedding facial."""
    return os.path.exists(verifier._emb_path(username))


def set_initial_funds(username: str, amount: float) -> None:
    data = {"username": username, "funds": float(amount)}
    with open(_user_json_path(username), "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def get_user_funds(username: str) -> float:
    path = _user_json_path(username)
    if not path.exists():
        return 0.0

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return float(data.get("funds", 0.0))


def transfer_funds(sender: str, receiver: str, amount: float) -> tuple[bool, str]:
    """Transfiere fondos si ambos usuarios existen y hay saldo suficiente."""
    sender_path = _user_json_path(sender)
    receiver_path = _user_json_path(receiver)

    if not sender_path.exists():
        return False, "El remitente no existe."
    if not receiver_path.exists():
        return False, "El destinatario no existe."

    with open(sender_path, "r", encoding="utf-8") as file:
        sender_data = json.load(file)
    with open(receiver_path, "r", encoding="utf-8") as file:
        receiver_data = json.load(file)

    sender_funds = float(sender_data.get("funds", 0))
    receiver_funds = float(receiver_data.get("funds", 0))

    if amount <= 0:
        return False, "La cantidad debe ser positiva."
    if sender_funds < amount:
        return False, "Fondos insuficientes."

    sender_data["funds"] = sender_funds - amount
    receiver_data["funds"] = receiver_funds + amount

    with open(sender_path, "w", encoding="utf-8") as file:
        json.dump(sender_data, file, indent=2)
    with open(receiver_path, "w", encoding="utf-8") as file:
        json.dump(receiver_data, file, indent=2)

    return True, f"Transferencia de {amount:.2f} EUR a {receiver} realizada correctamente."

from rai_audit.audit import run_audit


if __name__ == "__main__":
    outputs = run_audit()
    for name, value in outputs.items():
        print(f"{name}: {value}")

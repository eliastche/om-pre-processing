from pathlib import Path

p = Path(r"C:\Users\IFE13253\OneDrive - Institutt for Energiteknikk\Documents\OffshoreRisk\RiskSimulation\MonteCarlo-PostProcces\Results\Shadow power price by region - tech.csv")

print("Path given to pandas:")

print(p)


print("\nBasic checks:")
print("exists() :", p.exists())

print("is_file() :", p.is_file())

print("parent exists :", p.parent.exists())

print("cwd() :", Path.cwd())

print("resolved :", p.resolve(strict=False))

print("\nParent folder contents (matching files):")

if p.parent.exists():
    for x in p.parent.glob("*Shadow*power*price*tech*"):
        print(" -", x.name)
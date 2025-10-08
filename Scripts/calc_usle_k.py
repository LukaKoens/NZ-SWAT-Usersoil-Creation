import math

# # Prompt user for input values
# orgc = float(input("Enter organic carbon content (orgc): "))
# sand = float(input("Enter sand percentage: "))
# silt = float(input("Enter silt percentage: "))
# clay = float(input("Enter clay percentage: "))

def fcsand(sand, silt):
    """Calculate the fraction of sand."""
    a = (-0.256*sand*(1-(silt/100)))
    return (0.2 +0.3*math.exp(a))

def fcl_si(silt, clay):
    """Calculate the fraction of silt."""
    a = silt / (silt + clay)
    return pow(a, 0.3)

def forgc(orgc, sand):
    """Calculate the organic carbon factor."""
    a = orgc + math.exp(-5.51 + 22.9 * (1 - (sand / 100)))
    return 1 - (0.0256*orgc)/ a

def fhisand(sand):
    """Calculate the high sand factor."""
    a = 1 - sand / 100
    b = a + math.exp(-5.51 + 22.9 * a)
    return 1 - ((0.7*a)/b)

# K_USLE = fcsand(sand, silt) * fcl_si(silt, clay) * forgc(orgc, sand) * fhisand(sand)
# print(f"Calculated K_USLE value: {K_USLE}")

def calc_usle_k(orgc, sand, silt, clay):
    return fcsand(sand, silt) * fcl_si(silt, clay) * forgc(orgc, sand) * fhisand(sand)
    
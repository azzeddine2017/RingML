load "../src/ringml.ring"

see "=== Testing Dense Layer Initialization ===" + nl

# Test 1: Layer like the one in XOR (4 Inputs, 1 Output)
see "Creating Dense(4, 1)..." + nl
l = new Dense(4, 1)

see "Weights (Should be 4 rows, all non-zero floats):" + nl
l.oWeights.print()

# Check for zeros
nZeros = 0
for r = 1 to 4
    if l.oWeights.aData[r][1] = 0 
        nZeros++
        see "Row " + r + " is ZERO!" + nl
    ok
next

if nZeros = 0
    see "SUCCESS: No zeros found." + nl
else
    see "FAILURE: Found " + nZeros + " zero rows." + nl
ok

l = new Dense(4, 1)
see "Bias Row:" + nl
l.oBias.print()
import sympy as sy
x = sy.symbols('x')
y = x**3 + 10 + sy.sin(x)
dy_dx = sy.diff(y,x)
print(dy_dx)

int_y_x = sy.integrate(y,x)
print(int_y_x)
fix_int_y_x = sy.integrate(y,(x,0,1))
print(float(fix_int_y_x))

def QG_error(x,s):
    mu = 0      #均值
    sigma = 1   #标准差
    p_x = norm.pdf(x,mu,sigma)
    y1 = (s/127)*exp(abs(x))* p_x
    I1 = sy.integrate(y1,(x,(0,s)))
    y2 = (x-s)*exp(abs(x))* p_x
    I2 = sy.integrate(y2,(x,s,max(x)))

    return I1 + I2
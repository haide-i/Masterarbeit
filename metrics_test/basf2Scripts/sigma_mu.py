for i in range(1000):
    #print(i)
    exp_number = int(i)//10.
    #print(exp_number)
    sigma = exp_number // 10. * 0.2
    print(sigma)
    mu = exp_number % 10 * 0.5
    print(mu)

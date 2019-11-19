eps = 0.01 
df = 1
h = 0.7
omega_m = 0.286
omea_lambda = 0.714

def integrand(z):
    return 1/(2.269*(10**-18)*((1+z)*(np.sqrt(omega_m*(1+z)**3 + omega_lambda))))

def get_tlookback_0(scale):
    tlookback_0 = np.zeros(len(scale))
    for i in range (0,len(scale)):
        I = quad(integrand, 1/scale[0] - 1, 1/scale[i] - 1)
        tlookback_0[i] = I[0]
    return tlookback_0

def get_host_properties(tree,scalefactor):
    host_properties = {}
    host_properties['mvir'] = (1/h)*tree[np.in1d(tree['scale'], scalefactor, assume_unique=True)]['mvir']
    host_properties['vmax'] = (3.24*(10**-17))*tree[np.in1d(tree['scale'], scalefactor, assume_unique=True)]['vmax']
    host_properties['rvir'] = (1/h)*tree[np.in1d(tree['scale'], scalefactor, assume_unique=True)]['rvir']
    host_properties['rs'] = (1/h)*tree[np.in1d(tree['scale'], scalefactor, assume_unique=True)]['rs']
    host_properties['c'] = host_properties['rvir']/host_properties['rs']
    for coord in ['x', 'y', 'z']:
        host_properties['{}'.format(coord)] = (1000/h)*tree[np.in1d(tree['scale'], scalefactor, assume_unique=True)][coord]
    for vcoord in ['vx', 'vy', 'vz']:
        host_properties['{}'.format(vcoord)] = (3.24*(10**-17))*tree[np.in1d(tree['scale'], scalefactor, assume_unique=True)][vcoord]
    return host_properties

def get_subhalo_properties(destroyed_catalog,accretion_catalog,host_properties,j):
    sub_properties = {}
    sub_properties['mvir'] = (1/h)*destroyed_catalog[j][0]['mvir']
    sub_properties['rvir'] = (1/h)*destroyed_catalog[j][0]['rvir']
    sub_properties['rs'] = (1/h)*destroyed_catalog[j][0]['rs']
    sub_properties['vmax'] = destroyed_catalog[j][0]['vmax']
    sub_properties['xsub'] = rvir_sub/rs_sub
    sub_properties['macc'] = (1/h)*accretion_catalog[j][0][4]
    sub_properties['vacc'] = accretion_catalog[j][0][10]
    for coord in ['x', 'y', 'z']:
        sub_properties['{}'.format(coord)] = (1000/h)*destroyed_catalog[j][0][coord]
    for coord in ['vx', 'vy', 'vz']:
        sub_properties['{}'.format(vcoord)] = (3.24*(10**-17))*destroyed_catalog[j][0][vcoord]
    sub_properties['ri'] = np.sqrt((sub_properties['xi']-host_properties['x'])**2 + (sub_properties['yi']-host_properties['y'])**2 + (sub_properties['zi']-host_properties['z'])**2)
    return sub_properties

def get_Menc_NFW(mvir,c):
    return mvir*(np.log(1+(c*x))-(c*x)/(1+(c*x)))/(np.log(1+c)-c/(1+c))

def get_rho_NFW(Menc,x,c,rs_host):
    return Menc/((x*c*((1+x*c)**2))*4*np.pi*(np.log(1+c)-c/(1+c))*(rs_host)**3)

def get_sigma(vmax_host,x,c):
    return vmax_host*(1.4394*(c*x)**0.354)/(1+1.1756*(c*x)**0.725)

def get_scale_at_disruption(destroyed_catalog,j):
    scalefactor = destroyed_catalog[j][0][0]
    scalefactor_ind = np.argmin(np.abs(scale-scalefactor))
    return scalefactor, scalefactor_ind

# def get_I(xsub):
#     return (0.10947*((xsub)**3.989))/(1 + 0.90055*((xsub)**1.099) + 0.03568*((xsub)**1.189) + 0.06403*((xsub)**1.989))

def get_acceleration(Menc,host_properties,rho_host,sub_properties,eps,logLambda,Xratio):
    ax = (-df*G*Menc/((sub_properties['ri'] + eps*host_properties['rvir'])**2))*((sub_properties['x']-host_properties['x'])/sub_properties['ri'])
    ax -= 4*df*sub_properties['vx']*(np.pi)*logLambda*(G**2)*sub_properties['mvir']*rho_host*(scipy.special.erf(Xratio)-(2*Xratio*np.exp(-(Xratio**2)))/np.sqrt(np.pi))/(v**3)
    ay = (-df*G*Menc/((sub_properties['ri'] + eps*host_properties['rvir'])**2))*((sub_properties['y']-host_properties['y'])/sub_properties['ri'])
    ay -= 4*df*sub_properties['vy']*(np.pi)*logLambda*(G**2)*sub_properties['mvir']*rho_host*(scipy.special.erf(Xratio)-(2*Xratio*np.exp(-(Xratio**2)))/np.sqrt(np.pi))/(v**3)
    az = (-df*G*Menc/((sub_properties['ri'] + eps*host_properties['rvir'])**2))*((sub_properties['z']-host_properties['z'])/sub_properties['ri'])
    az -= 4*df*sub_properties['vz']*(np.pi)*logLambda*(G**2)*sub_properties['mvir']*rho_host*(scipy.special.erf(Xratio)-(2*Xratio*np.exp(-(Xratio**2)))/np.sqrt(np.pi))/(v**3)
    return ax, ay, az

def get_df_quantities(host_properties,sub_properties):
    #I = get_I(sub_properties['xsub'])
    mu_i = host_properties['mvir']/sub_properties['mvir']
    logLambda = np.log(mu_i)
    #logLambda = np.log(ri/rvir_sub) + I/((np.log(1+xsub)-xsub/(1+xsub))**2)
    x = sub_properties['ri']/(host_properties['rvir'])
    Menc = get_Menc_NFW(host_properties['mvir'],host_properties['c'])
    rho_host = get_rho_NFW(Menc,x,host_properties['c'],host_properties['rs'])
    sigma = get_sigma(host_properties['vmax'],x,host_properties['c'])
    v = np.sqrt((sub_properties['vx']-host_properties['vx'])**2 + (sub_properties['vy']-host_properties['vy'])**2 + (sub_properties['vz']-host_properties['vz'])**2)
    Xratio = v/(np.sqrt(2)*sigma)
    r = (G*host_properties['mvir'])/(v**2)
    eta = r/(host_properties['rvir'])
    l = LA.norm(np.cross(((sub_properties['x']-host_properties['x'])[0],(sub_properties['y']-host_properties['y'])[0],(sub_properties['z']-host_properties['z'])[0]),
        ((sub_properties['vx']-host_properties['vx'])[0],(sub_properties['vy']-host_properties['vy'])[0],(sub_properties['vz']-host_properties['vz'])[0])))
    epsilon = l/(r*v)
    #logLambda += 1.04*(((1/h)*tree[np.in1d(tree['scale'], scalefactor, assume_unique=True)]['mvir']/mvir_sub)**(0.64))*(eta**(0.72))*np.exp(-3.02*epsilon) - 0.75
    logLambda += 1.04*((mu_i)**(0.64))*(eta**(0.72))*np.exp(-3.02*epsilon) - 0.75
    rhovir = host_properties['mvir']/((4/3)*np.pi*(host_properties['rvir'])**3)
    tau = ((4/3)*np.pi*G*rhovir)**(-0.5)
    return logLambda, Menc, rho_host, sigma, Xratio, r, epsilon, rhovir, tau

def leapfrog(step,tlookback,deltat,sub_properties,host_properties,ax,ay,az,sigma,tau):
    for k in range (0,step):
            tlookback -= deltat
            sub_properties['x'] += (sub_properties['vx']*deltat + (ax*(deltat**2)/2))
            sub_properties['y'] += (sub_properties['vy']*deltat + (ay*(deltat**2)/2))
            sub_properties['z'] += (sub_properties['vz']*deltat + (az*(deltat**2)/2))
            host_properties['x'] += host_properties['vx']*deltat
            host_properties['y'] += host_properties['vy']*deltat
            host_properties['z'] += host_properties['vz']*deltat
            sub_properties['vx'] += ax*deltat
            sub_properties['vy'] += ay*deltat
            sub_properties['vz'] += az*deltat
            v = np.sqrt((sub_properties['vx']-host_properties['vx'])**2 + (sub_properties['vy']-host_properties['vy'])**2 + (sub_properties['vz']-host_properties['vz'])**2)
            Xratio = v/(np.sqrt(2)*sigma)
            r = (G*host_properties['mvir'])/(v**2)
            eta = r/(host_properties['rvir'])
            l = LA.norm(np.cross(((sub_properties['x']-host_properties['x'])[0],(sub_properties['y']-host_properties['y'])[0],(sub_properties['z']-host_properties['z'])[0]),
                ((sub_properties['vx']-host_properties['vx'])[0],(sub_properties['vy']-host_properties['vy'])[0],(sub_properties['vz']-host_properties['vz'])[0])))
            epsilon = l/(r*v)
            #logLambda += 1.04*(((1/h)*tree[np.in1d(tree['scale'], scalefactor, assume_unique=True)]['mvir']/mvir_sub)**(0.64))*(eta**(0.72))*np.exp(-3.02*epsilon) - 0.75
            logLambda += 1.04*(mu_i**(0.64))*(eta**(0.72))*np.exp(-3.02*epsilon) - 0.75
            ax, ay, az = get_acceleration(Menc,host_properties,rho_host,sub_properties,eps,logLambda,Xratio)
            if (np.sqrt((sub_properties['x']-host_properties['x'])**2 + (sub_properties['y']-host_properties['y'])**2 + (sub_properties['z']-host_properties['z'])**2) > sub_properties['ri']):
                delta_mvir = 1.18*sub_properties['mvir']*((sub_properties['mvir']/(host_properties['mvir'])**(0.07))/tau
                sub_properties['mvir'] -= delta_mvir
                sub_properties['vmax'] += delta_mvir*(sub_properties['vmax']/sub_properties['mvir'])*(0.3 - 0.4*sub_properties['mvir']/(sub_properties['mvir'] + sub_properties['macc']))
                test += 1
            sub_properties['ri'] = np.sqrt((sub_properties['x']-host_properties['x'])**2 + (sub_properties['y']-host_properties['y'])**2 + (sub_properties['z']-host_properties['z'])**2)
            logLambda = np.log(host_properties['mvir']/sub_properties['mvir'])
            #logLambda = np.log(ri/rvir_sub) + I/((np.log(1+xsub)-xsub/(1+xsub))**2)
            x = sub_properties['ri']/(host_properties['rvir'])
            Menc = get_Menc_NFW(host_properties['mvir'],host_properties['c'])
            rho_host = get_rho_NFW(Menc,x,host_properties['c'],host_properties['rs'])
            sigma = get_sigma(host_properties['vmax'],x,host_properties['c'])
    return sub_properties, host_properties, v, Xratio, r, eta, l, epsilon, logLambda, ax, ay, az, Menc, rho_host, sigma

def get_orphan_catalog(tree,destroyed_catalog,accretion_catalog,vpeak_test):
    tlookback_0 = get_tlookback_0(tree['scale'])
    for j in range(0,len(accretion_catalog)):
        #scale factor at disruption
        scalefactor, scalefactor_ind = get_scale_at_disruption(destroyed_catalog,j)
        atestext = np.zeros(scalefactor_ind); dtestext = np.zeros(scalefactor_ind);
        #host halo properties
        host_properties = get_host_properties(tree,scalefactor)
        #subhalo properties
        sub_properties = get_subhalo_properties(destroyed_catalog,accretion_catalog,host_properties,j)
        vpeak_sub = vpeak_test[j]
        #DF quantities
        logLambda, Menc, rho_host, sigma, Xratio, r, epsilon, rhovir, tau = get_df_quantities(host_properties,sub_properties)
        ax, ay, az = get_acceleration(Menc,host_properties,rho_host,sub_properties,eps,logLambda,Xratio)
        #Define output
        if scalefactor_ind == 0:
            orphan_catalog[j][0] = sub_properties['x']; orphan_catalog[j][1] = sub_properties['y']; orphan_catalog[j][2] = sub_properties['z']; orphan_catalog[j][3] = sub_properties['ri'];
            orphan_catalog[j][4] = sub_properties['vmax']; orphan_catalog[j][5] = vpeak_sub; orphan_catalog[j][6] = R200accvalues[j];
            orphan_catalog[j][7] = sub_properties['mvir']; orphan_catalog[j][8] = sub_properties['macc']; orphan_catalog[j][9] = sub_properties['vacc'];
            continue
        #Interpolate orbit
        i = 0; step = 10; tlookback_00 = tlookback_0[scalefactor_ind];
        for i in range (0,scalefactor_ind):
            scalefactor = tree['scale'][scalefactor_ind-i]
            scalefactor_next = tree['scale'][scalefactor_ind-i-1]
            scalefactor_ind_current = np.argmin(np.abs(scale-scalefactor))
            tlookback = tlookback_0[scalefactor_ind_current] - tlookback_0[scalefactor_ind_current-1]
            deltat = tlookback/step
            sub_properties, host_properties, v, Xratio, r, eta, l, epsilon, logLambda, ax, ay, az, Menc, rho_host, sigma = leapfrog(step,tlookback,deltat,sub_properties,host_properties,ax,ay,az,sigma,tau)
            host_properties = get_host_properties(tree,scalefactor_next)
            sub_properties['ri'] = np.sqrt((sub_properties['x']-host_properties['x'])**2 + (sub_properties['y']-host_properties['y'])**2 + (sub_properties['z']-host_properties['z'])**2)
            logLambda = np.log(host_properties['mvir']/sub_properties['mvir'])    
            atestext[i] = ((omega_m/omega_lambda)**(1/3))*(np.sinh(1.5*np.sqrt(omega_lambda)*2.269*(10**-18)*(tlookback_00-tlookback))**(2/3))
            dtestext[i] = ri
            #Write output
            orphan_catalog[j][0] = sub_properties['x']; orphan_catalog[j][1] = sub_properties['y']; orphan_catalog[j][2] = sub_properties['z']; orphan_catalog[j][3] = sub_properties['ri'];
            orphan_catalog[j][4] = sub_properties['vmax']; orphan_catalog[j][5] = vpeak_sub; orphan_catalog[j][6] = R200accvalues[j];
            orphan_catalog[j][7] = sub_properties['mvir']; orphan_catalog[j][8] = sub_properties['macc']; orphan_catalog[j][9] = sub_properties['vacc'];
            i += 1
    return np.asarray(orphan_catalog)
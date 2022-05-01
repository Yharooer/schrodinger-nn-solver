import numpy as np

class ResultsAgainstTimeSingleTest():
    def __init__(self, name, psi0_real_fn, psi0_imag_fn, v_fn, real_soln, imag_soln):
        self.psi0_real_fn = psi0_real_fn
        self.psi0_imag_fn = psi0_imag_fn
        self.real_soln = real_soln
        self.imag_soln = imag_soln
        self.v_fn = v_fn
        self.name = name

    def psi0_real(self, xs):
        return self.psi0_real_fn(xs)

    def psi0_imag(self, xs):
        return self.psi0_imag_fn(xs)

    def v(self, xs):
        return self.v_fn(xs)

    def real_soln(self, xs):
        return self.real_soln(xs)

    def imag_soln(self, xs):
        return self.imag_soln(xs)


class SingleTestCollectionFactory():
    def __init__(self):
        self.collection = []
    
    def add(self, name, psi0_real_fn, psi0_imag_fn, v_fn, real_soln, imag_soln):
        self.collection.append(ResultsAgainstTimeSingleTest(name, psi0_real_fn, psi0_imag_fn, v_fn, real_soln, imag_soln))


def get_tests():
    tests = SingleTestCollectionFactory()

    tests.add('particle_in_box_eigen1', lambda x: np.sqrt(2)*np.sin(np.pi*x), lambda x: 0*x, lambda x: 0*x, lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.cos((0.5 * np.pi**2)*t), lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.sin(-(0.5 * np.pi**2)*t))
    tests.add('particle_in_box_eigen2', lambda x: np.sqrt(2)*np.sin(2*np.pi*x), lambda x: 0*x, lambda x: 0*x, lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.cos((0.5 * 4 * np.pi**2)*t), lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.sin(-(0.5 * 4 * np.pi**2)*t))
    tests.add('particle_in_box_eigen3', lambda x: np.sqrt(2)*np.sin(3*np.pi*x), lambda x: 0*x, lambda x: 0*x, lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.cos((0.5 * 9 * np.pi**2)*t), lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.sin(-(0.5 * 9 * np.pi**2)*t))
    tests.add('particle_in_box_eigen4', lambda x: np.sqrt(2)*np.sin(4*np.pi*x), lambda x: 0*x, lambda x: 0*x, lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.cos((0.5 * 16 * np.pi**2)*t), lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.sin(-(0.5 * 16 * np.pi**2)*t))
    
    RAISED = 5
    tests.add('raised_box_eigen1', lambda x: np.sqrt(2)*np.sin(np.pi*x), lambda x: 0*x, lambda x: 0*x+RAISED, lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.cos((0.5 * np.pi**2 + RAISED)*t), lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.sin(-(0.5 * np.pi**2 + RAISED)*t))
    tests.add('raised_box_eigen2', lambda x: np.sqrt(2)*np.sin(2*np.pi*x), lambda x: 0*x, lambda x: 0*x+RAISED, lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.cos((0.5 * 4 * np.pi**2 + RAISED)*t), lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.sin(-(0.5 * 4 * np.pi**2 + RAISED)*t))
    tests.add('raised_box_eigen3', lambda x: np.sqrt(2)*np.sin(3*np.pi*x), lambda x: 0*x, lambda x: 0*x+RAISED, lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.cos((0.5 * 9 * np.pi**2 + RAISED)*t), lambda x,t: np.sqrt(2)*np.sin(np.pi*x)*np.sin(-(0.5 * 9 * np.pi**2 + RAISED)*t))

    return tests.collection
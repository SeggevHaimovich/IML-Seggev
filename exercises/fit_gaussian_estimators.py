from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    unigaus = UnivariateGaussian()
    mu = 10
    var = 1
    number_of_samples = 1000
    samples = np.random.normal(mu, var, number_of_samples)
    unigaus.fit(samples)
    print((unigaus.mu_, unigaus.var_))

    # Question 2 - Empirically showing sample mean is consistent
    expectations = []
    sample_num = np.arange(10, number_of_samples + 1, 10)
    for i in sample_num:
        unigaus.fit(samples[:i])
        expectations.append(unigaus.mu_)
    distance = np.abs(np.array(expectations) - mu)

    # plt.figure()
    # plt.plot(sample_num, distance)
    # plt.xlabel("number of samples")
    # plt.ylabel("r$distance - |\hat\mu - \mu|$")
    # plt.title("The estimated expectation error by number of samples")
    # plt.savefig('C:/Users/segge/source/repos/IML.HUJI/images/Ex1/firstplt.png')

    fig = go.Figure([go.Scatter(x=sample_num, y=distance, mode='lines')],
                    layout=go.Layout(title="The estimated expectation error by number of samples",
                                     xaxis_title="number of samples",
                                     yaxis_title="r$distance - |\hat\mu - \mu|$"))
    # pio.write_image(fig, "C:/Users/segge/source/repos/IML.HUJI/images/Ex1/firstgo.png")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model

    pdf_vec = unigaus.pdf(samples)

    # plt.figure()
    # plt.plot(samples, pdf_vec, '.')
    # plt.xlabel("Sample Value")
    # plt.ylabel("PDF Value")
    # plt.title("PDF by Sample value")
    # plt.savefig("C:/Users/segge/source/repos/IML.HUJI/images/Ex1/secondply.png")

    fig = go.Figure(data=[go.Scatter(x=samples, y=pdf_vec, mode='markers', name='what_is_this')],
                    layout=go.Layout(title="PDF by Sample value",
                                     xaxis_title="Sample Value",
                                     yaxis_title="PDF Value"))
    # fig.write_image("C:/Users/segge/source/repos/IML.HUJI/images/Ex1/secondgo.png")
    fig.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multigaus = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    cov_matrix = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    number_of_samples = 1000
    example = np.random.multivariate_normal(mu, cov_matrix, number_of_samples)
    multigaus.fit(example)
    print(multigaus.mu_)
    print(multigaus.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    values = np.ndarray((len(f1), len(f3)))
    max_value = -np.inf
    best_place = (0, 0)
    for i in range(len(f1)):
        for j in range(len(f3)):
            cur_mu = np.array([f1[i], 0, f3[j], 0])
            values[i][j] = MultivariateGaussian.log_likelihood(cur_mu, cov_matrix, example)
            if values[i][j] > max_value:
                max_value = values[i][j]
                best_place = (f1[i], f3[j])
    fig = go.Figure(go.Heatmap(x=f3, y=f1, z=values),
                    layout=go.Layout(title="log-likelihood of different f1 and f3 values - HEATMAP",
                                     xaxis_title="f3 value",
                                     yaxis_title="f1 value",
                                     height=500, width=500))
    # fig.write_image("C:/Users/segge/source/repos/IML.HUJI/images/Ex1/thirdgo.png")
    fig.show()

    # Question 6 - Maximum likelihood
    print("max log-liklihood is:", max_value)
    print("The maximum log-liklihood achived by:")
    print("f1:", round(best_place[0], 4))
    print("f3:", round(best_place[1], 4))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

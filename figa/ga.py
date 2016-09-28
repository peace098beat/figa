#!coding:utf-8
"""
総当り計算を行い最適パラメータを見つける
"""
import random

import tools


def eval_function(pop):
    pass


tools.mate = tools.cxOnePoint
tools.mutate = tools.mutUniformInt

# Init Individual Value
PMAX = 32
p11_enc, p11_dec, p11_init = tools.converter(min=1000, max=10000, n=PMAX, type=int)  # 0-9
p12_enc, p12_dec, p12_init = tools.converter(min=1000, max=8000, n=PMAX, type=int)  # 0-9
p21_enc, p21_dec, p21_init = tools.converter(min=1000, max=10000, n=PMAX, type=int)  # 0-9
p22_enc, p22_dec, p22_init = tools.converter(min=1000, max=8000, n=PMAX, type=int)  # 0-9
p31_enc, p31_dec, p31_init = tools.converter(min=1000, max=10000, n=PMAX, type=int)  # 0-9
p32_enc, p32_dec, p32_init = tools.converter(min=1000, max=8000, n=PMAX, type=int)  # 0-9

init_params = [p11_init, p12_init, p21_init, p22_init, p31_init, p32_init]
pdecoder = tools.decoder([p11_dec, p12_dec, p21_dec, p22_dec, p31_dec, p32_dec, ])

# mutation parameter
mut_low = [0] * 6
mut_up = [PMAX - 1] * 6
mut_indpb = 0.3



def main():
    random.seed(64)
    # 初期の個体群を生成
    NPOP = 100
    pop = tools.population(init_params, n=NPOP)
    CXPB, MUTPB, NGEN = 0.5, 0.02, 100  # 交差確率, 突然変異確率, 進化計算のループ(世代)

    print("Start of evolution")

    # 初期の個体群の評価
    fitnesses = list(map(tools.evalute, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness_values = fit

    print("Evaluted %i individuals" % len(pop))

    # 進化計算開始
    for g in range(NGEN):
        print("-- Generation {} --".format(g))

        # 時勢代の個体群を選択
        offspring = tools.selTournamentDCD(pop, NPOP)
        offspring = list(map(tools.clone, offspring))

        # 選択した個体に交叉と突然変位
        # 偶数版目と奇数版目の固体を取り出して交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                tools.mate(child1, child2)
                del child1.fitness_values, child1.fitness_valid
                del child2.fitness_values, child2.fitness_valid

        # 突然変異
        for mutant in offspring:
            if random.random() < MUTPB:
                tools.mutUniformInt(mutant, mut_low, mut_up, mut_indpb)
                del mutant.fitness_values, mutant.fitness_valid

        # 適合度を再計算
        invalid_ind = [ind for ind in offspring if not ind.fitness_valid]
        fitnesses = map(tools.evalute, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness_values = fit

        print("Evaluated {} individuals".format(len(invalid_ind)))

        # 次世代グンをoffspringにする
        pop[:] = tools.selBest(offspring + pop, NPOP)
        # pop[:] = tools.selTournamentDCD(offspring + pop, NPOP)

        # 全ての個体の適合度を配列にする
        fits = [ind.fitness_values for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        # print("  Min {}".format(min(fits)))
        # print("  Max {}".format(max(fits)))
        # print("  Avg {}".format(mean))
        # print("  Std {}".format(std))
        best_ind = tools.selBest(pop, 1)[0]
        report = "No.{}:{} : Best individual is {}, \n".format(
            g,
            best_ind.fitness_values,
            pdecoder(best_ind.x))

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is {}, {}".format(best_ind, best_ind.fitness_values))


if __name__ == '__main__':
    main()

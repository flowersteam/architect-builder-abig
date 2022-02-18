def save_training_graphs(train_recorder, save_dir):
    from alfred.utils.plots import plot_curves, create_fig
    import matplotlib.pyplot as plt

    # True Returns
    fig, axes = create_fig((1, 3))

    agg_interaction_step, agg_success = train_recorder.aggregate('interaction_step', 'success', 'mean', True)
    plot_curves(axes[0],
                ys=[agg_success],
                xs=[agg_interaction_step],
                xlabel="interaction_step", ylabel="success")
    plot_curves(axes[1],
                ys=[train_recorder.tape['manhattan_distance']],
                xs=[train_recorder.tape['step']],
                xlabel="step", ylabel="mean manhattan distance")
    plot_curves(axes[2],
                ys=[train_recorder.tape['distance_to_optimum']],
                xs=[train_recorder.tape['episode']],
                xlabel="episode", ylabel="episode-len / initial manhattan dist", markers='x', markevery=1)
    fig.savefig(str(save_dir / 'returns.png'))
    plt.close(fig)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def get_average_score_per_triplet(importance):
    avg_imp = np.zeros([importance.shape[0] + 2, 1])
    avg_imp[0:-2] += importance
    avg_imp[1:-1] += importance
    avg_imp[2:] += importance
    avg_imp[2:-2] = avg_imp[2:-2] / 3
    avg_imp[1] /= 2
    avg_imp[-2] /= 2
    avg_imp = avg_imp.tolist()
    avg_imp = [i[0] for i in avg_imp]

    return avg_imp

if __name__ == "__main__":

    sub = "P26358"
    save_path = "../results/importance/"
    sub_imp = pd.read_csv(save_path + sub + ".csv", names = [sub], delimiter=",")
    sub_imp = sub_imp.to_numpy()

    # The average score per triplet
    avg_imp = get_average_score_per_triplet(sub_imp)
    sub_seq = "MPARTAPARVPTLAVPAISLPDDVRRRLKDLERDSLTEKECVKEKLNLLHEFLQTEIKNQLCDLETKLRKEELSEEGYLAKVKSLLNKDLSLENGAHAYNREVNGRLENGNQARSEARRVGMADANSPPKPLSKPRTPRRSKSDGEAKPEPSPSPRITRKSTRQTTITSHFAKGPAKRKPQEESERAKSDESIKEEDKDQDEKRRRVTSRERVARPLPAEEPERAKSGTRTEKEEERDEKEEKRLRSQTKEPTPKQKLKEEPDREARAGVQADEDEDGDEKDEKKHRSQPKDLAAKRRPEEKEPEKVNPQISDEKDEDEKEEKRRKTTPKEPTEKKMARAKTVMNSKTHPPKCIQCGQYLDDPDLKYGQHPPDAVDEPQMLTNEKLSIFDANESGFESYEALPQHKLTCFSVYCKHGHLCPIDTGLIEKNIELFFSGSAKPIYDDDPSLEGGVNGKNLGPINEWWITGFDGGEKALIGFSTSFAEYILMDPSPEYAPIFGLMQEKIYISKIVVEFLQSNSDSTYEDLINKIETTVPPSGLNLNRFTEDSLLRHAQFVVEQVESYDEAGDSDEQPIFLTPCMRDLIKLAGVTLGQRRAQARRQTIRHSTREKDRGPTKATTTKLVYQIFDTFFAEQIEKDDREDKENAFKRRRCGVCEVCQQPECGKCKACKDMVKFGGSGRSKQACQERRCPNMAMKEADDDEEVDDNIPEMPSPKKMHQGKKKKQNKNRISWVGEAVKTDGKKSYYKKVCIDAETLEVGDCVSVIPDDSSKPLYLARVTALWEDSSNGQMFHAHWFCAGTDTVLGATSDPLELFLVDECEDMQLSYIHSKVKVIYKAPSENWAMEGGMDPESLLEGDDGKTYFYQLWYDQDYARFESPPKTQPTEDNKFKFCVSCARLAEMRQKEIPRVLEQLEDLDSRVLYYSATKNGILYRVGDGVYLPPEAFTFNIKLSSPVKRPRKEPVDEDLYPEHYRKYSDYIKGSNLDAPEPYRIGRIKEIFCPKKSNGRPNETDIKIRVNKFYRPENTHKSTPASYHADINLLYWSDEEAVVDFKAVQGRCTVEYGEDLPECVQVYSMGGPNRFYFLEAYNAKSKSFEDPPNHARSPGNKGKGKGKGKGKPKSQACEPSEPEIEIKLPKLRTLDVFSGCGGLSEGFHQAGISDTLWAIEMWDPAAQAFRLNNPGSTVFTEDCNILLKLVMAGETTNSRGQRLPQKGDVEMLCGGPPCQGFSGMNRFNSRTYSKFKNSLVVSFLSYCDYYRPRFFLLENVRNFVSFKRSMVLKLTLRCLVRMGYQCTFGVLQAGQYGVAQTRRRAIILAAAPGEKLPLFPEPLHVFAPRACQLSVVVDDKKFVSNITRLSSGPFRTITVRDTMSDLPEVRNGASALEISYNGEPQSWFQRQLRGAQYQPILRDHICKDMSALVAARMRHIPLAPGSDWRDLPNIEVRLSDGTMARKLRYTHHDRKNGRSSSGALRGVCSCVEAGKACDPAARQFNTLIPWCLPHTGNRHNHWAGLYGRLEWDGFFSTTVTNPEPMGKQGRVLHPEQHRVVSVRECARSQGFPDTYRLFGNILDKHRQVGNAVPPPLAKAIGLEIKLCMLAKARESASAKIKEEEAAKD"
    start = 940
    end = 1300
    avg_imp = avg_imp[start-1:end-1]
    sub_seq = sub_seq[start-1:end-1]

    # def draw(pandas_data, seque, v_max, output_name_1, output_name_2):
    prtein_len = len(sub_seq)

    if (prtein_len / 20) > (np.floor(prtein_len / 20)):
        row_number = int(np.floor(prtein_len / 20) + 1)
    else:
        row_number = int(np.floor(prtein_len / 20))

    row_index = []
    for n in range(start, end, 20):
        row_index.append(int(n))

    def seq_mapping(row_number, seque, type):
        xx = np.zeros((row_number, 20))
        xx = xx.astype(type)
        ik = 0
        for n1 in range(row_number):
            for n2 in range(20):
                if ik < len(seque):
                    xx[n1][n2] = seque[ik]
                    ik += 1
        return xx

    seq_matrix = seq_mapping(row_number, sub_seq, type = str)
    imp_matrix = seq_mapping(row_number, avg_imp, type = np.float32)

    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=(0.667 * row_number, 12))
    sns.set_style('white')
    cdict = [(0, '#337071'), (0.5, '#FFFFFF'), (1, '#B51D23')]
    col = LinearSegmentedColormap.from_list('', cdict)

    sns.heatmap(data=imp_matrix, linewidths=0.1, annot=seq_matrix, annot_kws={'fontsize': 15}, fmt='', linecolor='#DCDCDE',
                 ax=ax, cmap=col, yticklabels=1, square=True,
                cbar=True , cbar_kws={'shrink':0.5, 'aspect':10, 'pad':0.08})

    ax.set_ylim([row_number, 0])
    ax.set_facecolor('#DCDCDE')
    ax.set_yticklabels(row_index, rotation=360)

    ax.tick_params(bottom=False, labelbottom=False)
    for edge in ['top', 'bottom', 'left', 'right']:
        ax.spines[edge].set_visible(True)
        ax.spines[edge].set_color('black')

    ax.tick_params(axis='y', labelsize=16, pad=0.5)

    visualization_name = sub + '_importance_map.png'
    fig.savefig(save_path + visualization_name, bbox_inches='tight', dpi=300, pad_inches=0.0)

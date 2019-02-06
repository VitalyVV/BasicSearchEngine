import glob
import importlib
import os
import sys
import lab3
from collections import Counter
from math import isclose


class HidePrints:  # helper
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


dir_name = "test_dir"
os.chdir(dir_name + "/")
# for each solution in the folder
for file in glob.glob("*.py"):
    print(file)
    ok = '\033[92mOK\033[0m'
    fail = '\033[91mFAIL\033[0m'
    module = importlib.import_module(dir_name + '.' + file[:-3])

    # test generate_wildcard_options
    dictionary = {'lorem': 1, 'ipsum': 1, 'dolor': 2, 'sit': 1, 'amet': 1, 'consectetur': 1, 'adipiscing': 1, 'elit': 1, 'sed': 1, 'do': 1, 'eiusmod': 1, 'tempor': 1, 'incididunt': 1, 'ut': 3, 'labore': 1, 'et': 1, 'dolore': 2, 'magna': 1, 'aliqua': 1, 'enim': 1, 'ad': 1, 'minim': 1, 'veniam': 1, 'quis': 1, 'nostrud': 1, 'exercitation': 1, 'ullamco': 1, 'laboris': 1, 'nisi': 1, 'aliquip': 1, 'ex': 1, 'ea': 1, 'commodo': 1, 'consequat': 1, 'duis': 1, 'aute': 1, 'irure': 1, 'reprehenderit': 1, 'voluptate': 1, 'velit': 1, 'esse': 1, 'cillum': 1, 'eu': 1, 'fugiat': 1, 'nulla': 1, 'pariatur': 1, 'excepteur': 1, 'sint': 1, 'occaecat': 1, 'cupidatat': 1, 'non': 1, 'proident': 1, 'sunt': 1, 'culpa': 1, 'qui': 1, 'officia': 1, 'deserunt': 1, 'mollit': 1, 'anim': 1, 'id': 1, 'est': 1, 'laborum': 1}
    k2_gram_index = {'$l': ['lorem', 'labore', 'laboris', 'laborum'], 'lo': ['lorem', 'dolor', 'dolore'], 'or': ['lorem', 'dolor', 'tempor', 'labore', 'dolore', 'laboris', 'laborum'], 're': ['lorem', 'labore', 'dolore', 'irure', 'reprehenderit', 'reprehenderit'], 'em': ['lorem', 'tempor'], 'm$': ['lorem', 'ipsum', 'enim', 'minim', 'veniam', 'cillum', 'anim', 'laborum'], '$i': ['ipsum', 'incididunt', 'irure', 'id'], 'ip': ['ipsum', 'adipiscing', 'aliquip'], 'ps': ['ipsum'], 'su': ['ipsum', 'sunt'], 'um': ['ipsum', 'cillum', 'laborum'], '$d': ['dolor', 'do', 'dolore', 'duis', 'deserunt'], 'do': ['dolor', 'do', 'dolore', 'commodo'], 'ol': ['dolor', 'dolore', 'voluptate', 'mollit'], 'r$': ['dolor', 'consectetur', 'tempor', 'pariatur', 'excepteur'], '$s': ['sit', 'sed', 'sint', 'sunt'], 'si': ['sit', 'nisi', 'sint'], 'it': ['sit', 'elit', 'exercitation', 'reprehenderit', 'velit', 'mollit'], 't$': ['sit', 'amet', 'elit', 'incididunt', 'ut', 'et', 'consequat', 'reprehenderit', 'velit', 'fugiat', 'sint', 'occaecat', 'cupidatat', 'proident', 'sunt', 'deserunt', 'mollit', 'est'], '$a': ['amet', 'adipiscing', 'aliqua', 'ad', 'aliquip', 'aute', 'anim'], 'am': ['amet', 'veniam', 'ullamco'], 'me': ['amet'], 'et': ['amet', 'consectetur', 'et'], '$c': ['consectetur', 'commodo', 'consequat', 'cillum', 'cupidatat', 'culpa'], 'co': ['consectetur', 'ullamco', 'commodo', 'consequat'], 'on': ['consectetur', 'exercitation', 'consequat', 'non'], 'ns': ['consectetur', 'consequat'], 'se': ['consectetur', 'sed', 'consequat', 'esse', 'deserunt'], 'ec': ['consectetur', 'occaecat'], 'ct': ['consectetur'], 'te': ['consectetur', 'tempor', 'aute', 'voluptate', 'excepteur'], 'tu': ['consectetur', 'pariatur'], 'ur': ['consectetur', 'irure', 'pariatur', 'excepteur'], 'ad': ['adipiscing', 'ad'], 'di': ['adipiscing', 'incididunt'], 'pi': ['adipiscing', 'cupidatat'], 'is': ['adipiscing', 'quis', 'laboris', 'nisi', 'duis'], 'sc': ['adipiscing'], 'ci': ['adipiscing', 'incididunt', 'exercitation', 'cillum', 'officia'], 'in': ['adipiscing', 'incididunt', 'minim', 'sint'], 'ng': ['adipiscing'], 'g$': ['adipiscing'], '$e': ['elit', 'eiusmod', 'et', 'enim', 'exercitation', 'ex', 'ea', 'esse', 'eu', 'excepteur', 'est'], 'el': ['elit', 'velit'], 'li': ['elit', 'aliqua', 'aliquip', 'velit', 'mollit'], 'ed': ['sed'], 'd$': ['sed', 'eiusmod', 'ad', 'nostrud', 'id'], 'o$': ['do', 'ullamco', 'commodo'], 'ei': ['eiusmod'], 'iu': ['eiusmod'], 'us': ['eiusmod'], 'sm': ['eiusmod'], 'mo': ['eiusmod', 'commodo', 'mollit'], 'od': ['eiusmod', 'commodo'], '$t': ['tempor'], 'mp': ['tempor'], 'po': ['tempor'], 'nc': ['incididunt'], 'id': ['incididunt', 'incididunt', 'cupidatat', 'proident', 'id'], 'du': ['incididunt', 'duis'], 'un': ['incididunt', 'sunt', 'deserunt'], 'nt': ['incididunt', 'sint', 'proident', 'sunt', 'deserunt'], '$u': ['ut', 'ullamco'], 'ut': ['ut', 'aute'], 'la': ['labore', 'ullamco', 'laboris', 'nulla', 'laborum'], 'ab': ['labore', 'laboris', 'laborum'], 'bo': ['labore', 'laboris', 'laborum'], 'e$': ['labore', 'dolore', 'aute', 'irure', 'voluptate', 'esse'], '$m': ['magna', 'minim', 'mollit'], 'ma': ['magna'], 'ag': ['magna'], 'gn': ['magna'], 'na': ['magna'], 'a$': ['magna', 'aliqua', 'ea', 'nulla', 'culpa', 'officia'], 'al': ['aliqua', 'aliquip'], 'iq': ['aliqua', 'aliquip'], 'qu': ['aliqua', 'quis', 'aliquip', 'consequat', 'qui'], 'ua': ['aliqua', 'consequat'], 'en': ['enim', 'veniam', 'reprehenderit', 'proident'], 'ni': ['enim', 'minim', 'veniam', 'nisi', 'anim'], 'im': ['enim', 'minim', 'anim'], 'mi': ['minim'], '$v': ['veniam', 'voluptate', 'velit'], 've': ['veniam', 'velit'], 'ia': ['veniam', 'fugiat', 'pariatur', 'officia'], '$q': ['quis', 'qui'], 'ui': ['quis', 'aliquip', 'duis', 'qui'], 's$': ['quis', 'laboris', 'duis'], '$n': ['nostrud', 'nisi', 'nulla', 'non'], 'no': ['nostrud', 'non'], 'os': ['nostrud'], 'st': ['nostrud', 'est'], 'tr': ['nostrud'], 'ru': ['nostrud', 'irure', 'deserunt', 'laborum'], 'ud': ['nostrud'], 'ex': ['exercitation', 'ex', 'excepteur'], 'xe': ['exercitation'], 'er': ['exercitation', 'reprehenderit', 'deserunt'], 'rc': ['exercitation'], 'ta': ['exercitation', 'voluptate', 'cupidatat'], 'at': ['exercitation', 'consequat', 'voluptate', 'fugiat', 'pariatur', 'occaecat', 'cupidatat', 'cupidatat'], 'ti': ['exercitation'], 'io': ['exercitation'], 'n$': ['exercitation', 'non'], 'ul': ['ullamco', 'nulla', 'culpa'], 'll': ['ullamco', 'cillum', 'nulla', 'mollit'], 'mc': ['ullamco'], 'ri': ['laboris', 'reprehenderit', 'pariatur'], 'i$': ['nisi', 'qui'], 'p$': ['aliquip'], 'x$': ['ex'], 'ea': ['ea'], 'om': ['commodo'], 'mm': ['commodo'], 'eq': ['consequat'], 'au': ['aute'], 'ir': ['irure'], '$r': ['reprehenderit'], 'ep': ['reprehenderit', 'excepteur'], 'pr': ['reprehenderit', 'proident'], 'eh': ['reprehenderit'], 'he': ['reprehenderit'], 'nd': ['reprehenderit'], 'de': ['reprehenderit', 'proident', 'deserunt'], 'vo': ['voluptate'], 'lu': ['voluptate', 'cillum'], 'up': ['voluptate', 'cupidatat'], 'pt': ['voluptate', 'excepteur'], 'es': ['esse', 'deserunt', 'est'], 'ss': ['esse'], 'il': ['cillum'], 'eu': ['eu', 'excepteur'], 'u$': ['eu'], '$f': ['fugiat'], 'fu': ['fugiat'], 'ug': ['fugiat'], 'gi': ['fugiat'], 'nu': ['nulla'], '$p': ['pariatur', 'proident'], 'pa': ['pariatur', 'culpa'], 'ar': ['pariatur'], 'xc': ['excepteur'], 'ce': ['excepteur'], '$o': ['occaecat', 'officia'], 'oc': ['occaecat'], 'cc': ['occaecat'], 'ca': ['occaecat', 'occaecat'], 'ae': ['occaecat'], 'cu': ['cupidatat', 'culpa'], 'da': ['cupidatat'], 'ro': ['proident'], 'oi': ['proident'], 'lp': ['culpa'], 'of': ['officia'], 'ff': ['officia'], 'fi': ['officia'], 'ic': ['officia'], 'an': ['anim']}
    k3_gram_index = {'$lo': ['lorem'], 'lor': ['lorem', 'dolor', 'dolore'], 'ore': ['lorem', 'labore', 'dolore'], 'rem': ['lorem'], 'em$': ['lorem'], '$ip': ['ipsum'], 'ips': ['ipsum'], 'psu': ['ipsum'], 'sum': ['ipsum'], 'um$': ['ipsum', 'cillum', 'laborum'], '$do': ['dolor', 'do', 'dolore'], 'dol': ['dolor', 'dolore'], 'olo': ['dolor', 'dolore'], 'or$': ['dolor', 'tempor'], '$si': ['sit', 'sint'], 'sit': ['sit'], 'it$': ['sit', 'elit', 'reprehenderit', 'velit', 'mollit'], '$am': ['amet'], 'ame': ['amet'], 'met': ['amet'], 'et$': ['amet', 'et'], '$co': ['consectetur', 'commodo', 'consequat'], 'con': ['consectetur', 'consequat'], 'ons': ['consectetur', 'consequat'], 'nse': ['consectetur', 'consequat'], 'sec': ['consectetur'], 'ect': ['consectetur'], 'cte': ['consectetur'], 'tet': ['consectetur'], 'etu': ['consectetur'], 'tur': ['consectetur', 'pariatur'], 'ur$': ['consectetur', 'pariatur', 'excepteur'], '$ad': ['adipiscing', 'ad'], 'adi': ['adipiscing'], 'dip': ['adipiscing'], 'ipi': ['adipiscing'], 'pis': ['adipiscing'], 'isc': ['adipiscing'], 'sci': ['adipiscing'], 'cin': ['adipiscing'], 'ing': ['adipiscing'], 'ng$': ['adipiscing'], '$el': ['elit'], 'eli': ['elit', 'velit'], 'lit': ['elit', 'velit', 'mollit'], '$se': ['sed'], 'sed': ['sed'], 'ed$': ['sed'], 'do$': ['do', 'commodo'], '$ei': ['eiusmod'], 'eiu': ['eiusmod'], 'ius': ['eiusmod'], 'usm': ['eiusmod'], 'smo': ['eiusmod'], 'mod': ['eiusmod', 'commodo'], 'od$': ['eiusmod'], '$te': ['tempor'], 'tem': ['tempor'], 'emp': ['tempor'], 'mpo': ['tempor'], 'por': ['tempor'], '$in': ['incididunt'], 'inc': ['incididunt'], 'nci': ['incididunt'], 'cid': ['incididunt'], 'idi': ['incididunt'], 'did': ['incididunt'], 'idu': ['incididunt'], 'dun': ['incididunt'], 'unt': ['incididunt', 'sunt', 'deserunt'], 'nt$': ['incididunt', 'sint', 'proident', 'sunt', 'deserunt'], '$ut': ['ut'], 'ut$': ['ut'], '$la': ['labore', 'laboris', 'laborum'], 'lab': ['labore', 'laboris', 'laborum'], 'abo': ['labore', 'laboris', 'laborum'], 'bor': ['labore', 'laboris', 'laborum'], 're$': ['labore', 'dolore', 'irure'], '$et': ['et'], '$ma': ['magna'], 'mag': ['magna'], 'agn': ['magna'], 'gna': ['magna'], 'na$': ['magna'], '$al': ['aliqua', 'aliquip'], 'ali': ['aliqua', 'aliquip'], 'liq': ['aliqua', 'aliquip'], 'iqu': ['aliqua', 'aliquip'], 'qua': ['aliqua', 'consequat'], 'ua$': ['aliqua'], '$en': ['enim'], 'eni': ['enim', 'veniam'], 'nim': ['enim', 'minim', 'anim'], 'im$': ['enim', 'minim', 'anim'], 'ad$': ['ad'], '$mi': ['minim'], 'min': ['minim'], 'ini': ['minim'], '$ve': ['veniam', 'velit'], 'ven': ['veniam'], 'nia': ['veniam'], 'iam': ['veniam'], 'am$': ['veniam'], '$qu': ['quis', 'qui'], 'qui': ['quis', 'aliquip', 'qui'], 'uis': ['quis', 'duis'], 'is$': ['quis', 'laboris', 'duis'], '$no': ['nostrud', 'non'], 'nos': ['nostrud'], 'ost': ['nostrud'], 'str': ['nostrud'], 'tru': ['nostrud'], 'rud': ['nostrud'], 'ud$': ['nostrud'], '$ex': ['exercitation', 'ex', 'excepteur'], 'exe': ['exercitation'], 'xer': ['exercitation'], 'erc': ['exercitation'], 'rci': ['exercitation'], 'cit': ['exercitation'], 'ita': ['exercitation'], 'tat': ['exercitation', 'voluptate', 'cupidatat'], 'ati': ['exercitation'], 'tio': ['exercitation'], 'ion': ['exercitation'], 'on$': ['exercitation', 'non'], '$ul': ['ullamco'], 'ull': ['ullamco', 'nulla'], 'lla': ['ullamco', 'nulla'], 'lam': ['ullamco'], 'amc': ['ullamco'], 'mco': ['ullamco'], 'co$': ['ullamco'], 'ori': ['laboris'], 'ris': ['laboris'], '$ni': ['nisi'], 'nis': ['nisi'], 'isi': ['nisi'], 'si$': ['nisi'], 'uip': ['aliquip'], 'ip$': ['aliquip'], 'ex$': ['ex'], '$ea': ['ea'], 'ea$': ['ea'], 'com': ['commodo'], 'omm': ['commodo'], 'mmo': ['commodo'], 'odo': ['commodo'], 'seq': ['consequat'], 'equ': ['consequat'], 'uat': ['consequat'], 'at$': ['consequat', 'fugiat', 'occaecat', 'cupidatat'], '$du': ['duis'], 'dui': ['duis'], '$au': ['aute'], 'aut': ['aute'], 'ute': ['aute'], 'te$': ['aute', 'voluptate'], '$ir': ['irure'], 'iru': ['irure'], 'rur': ['irure'], 'ure': ['irure'], '$re': ['reprehenderit'], 'rep': ['reprehenderit'], 'epr': ['reprehenderit'], 'pre': ['reprehenderit'], 'reh': ['reprehenderit'], 'ehe': ['reprehenderit'], 'hen': ['reprehenderit'], 'end': ['reprehenderit'], 'nde': ['reprehenderit'], 'der': ['reprehenderit'], 'eri': ['reprehenderit'], 'rit': ['reprehenderit'], '$vo': ['voluptate'], 'vol': ['voluptate'], 'olu': ['voluptate'], 'lup': ['voluptate'], 'upt': ['voluptate'], 'pta': ['voluptate'], 'ate': ['voluptate'], 'vel': ['velit'], '$es': ['esse', 'est'], 'ess': ['esse'], 'sse': ['esse'], 'se$': ['esse'], '$ci': ['cillum'], 'cil': ['cillum'], 'ill': ['cillum'], 'llu': ['cillum'], 'lum': ['cillum'], '$eu': ['eu'], 'eu$': ['eu'], '$fu': ['fugiat'], 'fug': ['fugiat'], 'ugi': ['fugiat'], 'gia': ['fugiat'], 'iat': ['fugiat', 'pariatur'], '$nu': ['nulla'], 'nul': ['nulla'], 'la$': ['nulla'], '$pa': ['pariatur'], 'par': ['pariatur'], 'ari': ['pariatur'], 'ria': ['pariatur'], 'atu': ['pariatur'], 'exc': ['excepteur'], 'xce': ['excepteur'], 'cep': ['excepteur'], 'ept': ['excepteur'], 'pte': ['excepteur'], 'teu': ['excepteur'], 'eur': ['excepteur'], 'sin': ['sint'], 'int': ['sint'], '$oc': ['occaecat'], 'occ': ['occaecat'], 'cca': ['occaecat'], 'cae': ['occaecat'], 'aec': ['occaecat'], 'eca': ['occaecat'], 'cat': ['occaecat'], '$cu': ['cupidatat', 'culpa'], 'cup': ['cupidatat'], 'upi': ['cupidatat'], 'pid': ['cupidatat'], 'ida': ['cupidatat'], 'dat': ['cupidatat'], 'ata': ['cupidatat'], 'non': ['non'], '$pr': ['proident'], 'pro': ['proident'], 'roi': ['proident'], 'oid': ['proident'], 'ide': ['proident'], 'den': ['proident'], 'ent': ['proident'], '$su': ['sunt'], 'sun': ['sunt'], 'cul': ['culpa'], 'ulp': ['culpa'], 'lpa': ['culpa'], 'pa$': ['culpa'], 'ui$': ['qui'], '$of': ['officia'], 'off': ['officia'], 'ffi': ['officia'], 'fic': ['officia'], 'ici': ['officia'], 'cia': ['officia'], 'ia$': ['officia'], '$de': ['deserunt'], 'des': ['deserunt'], 'ese': ['deserunt'], 'ser': ['deserunt'], 'eru': ['deserunt'], 'run': ['deserunt'], '$mo': ['mollit'], 'mol': ['mollit'], 'oll': ['mollit'], 'lli': ['mollit'], '$an': ['anim'], 'ani': ['anim'], '$id': ['id'], 'id$': ['id'], 'est': ['est'], 'st$': ['est'], 'oru': ['laborum'], 'rum': ['laborum']}
    wc_test = {"*ore": ['dolore', 'labore'],
               "*qui*": ['aliquip', 'qui', 'quis'],
               "*up*ta*": ['cupidatat', 'voluptate']}
    passed = True
    for pair in wc_test.items():
        with HidePrints():
            options = module.generate_wildcard_options(pair[0], k2_gram_index)
            options.sort()
        if options != pair[1]:
            passed = False
    print("wildcard", ok if passed else fail)

    # test produce_soundex_code
    soundex_test = {"britney": "b635",
                    "britain": "b635",
                    "priteny": "p635",
                    "retrieval": "r361",
                    "ritrivl": "r361",
                    "lorem": "l650",
                    "lorrrremmn": "l650",
                    "awe": "a000"}
    passed = True
    for pair in soundex_test.items():
        with HidePrints():
            code = module.produce_soundex_code(pair[0])
            if code != pair[1]:
                passed = False
    print("soundex", ok if passed else fail)
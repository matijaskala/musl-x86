#include <wchar.h>

wchar_t *wcscpy(wchar_t *restrict d, const wchar_t *restrict s)
{
	wcpcpy(d, s);
	return d;
}
